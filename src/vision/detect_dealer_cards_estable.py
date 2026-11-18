"""Detector experimental estable: Otsu + umbral respaldo + Canny, con diagnostico.
No usa PyTorch ni altera el dataset. Ejecutar desde src/vision/.
"""
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
import detect_dealer_cards as original

W, H = original.CARD_WIDTH, original.CARD_HEIGHT
MAX = original.MAX_CARDS


def ordenar(puntos):
    p = np.asarray(puntos, np.float32).reshape(4, 2)
    centro = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1]-centro[1], p[:, 0]-centro[0]))]
    p = np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)
    v1, v2 = p[1]-p[0], p[2]-p[1]
    if v1[0]*v2[1]-v1[1]*v2[0] < 0:
        p = p[[0, 3, 2, 1]]
    return p


def vertices(contorno):
    hull = cv2.convexHull(contorno)
    per = cv2.arcLength(hull, True)
    for eps in (0.012, 0.016, 0.02, 0.025, 0.03, 0.04, 0.05):
        poly = cv2.approxPolyDP(hull, eps*per, True)
        if len(poly) == 4 and cv2.isContourConvex(poly):
            p = ordenar(poly)
            if min(np.linalg.norm(p[(i+1)%4]-p[i]) for i in range(4)) > 12:
                return p, '4 vertices'
    return ordenar(cv2.boxPoints(cv2.minAreaRect(contorno))), 'rectangulo'


def candidatos(imagen, mascara, metodo):
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_imagen = imagen.shape[0]*imagen.shape[1]
    validos, descartados = [], []
    for c in contornos:
        area = cv2.contourArea(c)
        if area < original.MIN_CARD_AREA * 0.75:
            continue
        rect = cv2.minAreaRect(c)
        a, b = rect[1]
        if min(a, b) < 10:
            continue
        razon = max(a,b)/min(a,b)
        relleno = area/(a*b)
        motivo = None
        if area > area_imagen*.95:
            motivo = 'muy grande'
        elif not 1.10 <= razon <= 2.15:
            motivo = f'ratio {razon:.2f}'
        elif relleno < .60:
            motivo = f'relleno {relleno:.2f}'
        if motivo:
            descartados.append((c, f'{metodo}: {motivo}'))
            continue
        p, forma = vertices(c)
        validos.append((p, f'{metodo} {forma}', area))
    return validos, descartados


def iou_poligonos(a, b, forma):
    ma = np.zeros(forma[:2], np.uint8)
    mb = np.zeros(forma[:2], np.uint8)
    cv2.fillConvexPoly(ma, a.astype(np.int32), 1)
    cv2.fillConvexPoly(mb, b.astype(np.int32), 1)
    inter = np.count_nonzero(ma & mb)
    union = np.count_nonzero(ma | mb)
    return inter/union if union else 0


def detectar(dealer):
    _, otsu = original.make_binary(dealer)
    gris = cv2.cvtColor(dealer, cv2.COLOR_BGR2GRAY)
    _, fijo = cv2.threshold(gris, 135, 255, cv2.THRESH_BINARY)
    fijo = cv2.morphologyEx(fijo, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    suave = cv2.GaussianBlur(gris, (5,5), 0)
    bordes = cv2.Canny(suave, 45, 125)
    bordes = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    bordes = cv2.dilate(bordes, np.ones((3,3), np.uint8), iterations=1)
    resultados, rechazados = [], []
    # Se prioriza el metodo original; los otros solo recuperan candidatos faltantes.
    for mascara, nombre in ((otsu, 'Otsu'), (fijo, 'Respaldo'), (bordes, 'Canny')):
        encontrados, malos = candidatos(dealer, mascara, nombre)
        rechazados.extend(malos)
        for p, etiqueta, area in sorted(encontrados, key=lambda item: -item[2]):
            if any(iou_poligonos(p, anterior[0], dealer.shape) > .38 for anterior in resultados):
                continue
            resultados.append((p, etiqueta))
    resultados.sort(key=lambda x: x[0][:,0].mean())
    return resultados[:MAX], rechazados, {'Otsu':otsu,'Respaldo':fijo,'Canny':bordes}


def enderezar(imagen, pts, margen=.025):
    mx,my=W*margen,H*margen
    destino=np.array([[mx,my],[W-1-mx,my],[W-1-mx,H-1-my],[mx,H-1-my]],np.float32)
    return cv2.warpPerspective(imagen,cv2.getPerspectiveTransform(ordenar(pts),destino),(W,H))


def mosaico(imagenes, nombre):
    if not imagenes:
        return np.zeros((270,190,3),np.uint8)
    paneles=[]
    for i, img in enumerate(imagenes,1):
        panel=np.full((270,190,3),45,np.uint8)
        h,w=img.shape[:2]
        factor=min(178/w,234/h)
        im=cv2.resize(img,(max(1,round(w*factor)),max(1,round(h*factor))))
        panel[29:29+im.shape[0],6:6+im.shape[1]]=im
        cv2.putText(panel,f'{nombre} {i}',(7,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        paneles.append(panel)
    return np.hstack(paneles)


def main():
    cfg=original.cam_cfg
    cam=cv2.VideoCapture(cfg['index'])
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,cfg['width'])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,cfg['height'])
    if not cam.isOpened():
        raise SystemExit('No se pudo abrir la camara. Cierra otros programas que la usen.')
    historial=deque(maxlen=300)
    diagnostico=False
    print('DETECTOR ESTABLE | Q salir | D diagnostico | S guardar | R reiniciar medicion')
    print('La estabilidad mide cuadros con 4 detecciones; no garantiza identidad correcta.')
    ultimo=None
    try:
        while True:
            ok,frame=cam.read()
            if not ok: break
            h,w=frame.shape[:2]
            x1,y1,x2,y2=map(int,original.zones['DEALER'])
            x1,x2=sorted((max(0,min(w,x1)),max(0,min(w,x2))))
            y1,y2=sorted((max(0,min(h,y1)),max(0,min(h,y2))))
            if x2<=x1 or y2<=y1: raise ValueError('Zona DEALER invalida')
            dealer=frame[y1:y2,x1:x2].copy()
            cartas,rechazados,mascaras=detectar(dealer)
            historial.append(len(cartas)==MAX)
            estabilidad=100*sum(historial)/len(historial)
            vista=dealer.copy()
            corregidas=[]; indices=[]
            for i,(pts,metodo) in enumerate(cartas,1):
                cv2.polylines(vista,[pts.astype(np.int32)],True,(0,255,0),2)
                for p in pts: cv2.circle(vista,tuple(p.astype(int)),3,(0,0,255),-1)
                px,py=pts[0].astype(int)
                cv2.putText(vista,f'{i} {metodo}',(max(0,px),max(38,py-5)),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,255,255),1)
                carta=enderezar(dealer,pts)
                corregidas.append(carta)
                indices.append(carta[:int(H*.49),:int(W*.27)])
            cv2.putText(vista,f'Cartas: {len(cartas)}/{MAX} | Estabilidad: {estabilidad:.1f}% ({len(historial)} frames)',
                        (8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,255,255),1)
            completa=frame.copy()
            cv2.rectangle(completa,(x1,y1),(x2-1,y2-1),(255,255,0),2)
            cv2.imshow('ESTABLE - CONTORNOS',vista)
            cv2.imshow('ESTABLE - CARTAS CORREGIDAS',mosaico(corregidas,'Carta'))
            cv2.imshow('ESTABLE - INDICES 27%',mosaico(indices,'27%'))
            cv2.imshow('ESTABLE - CAMARA Y ZONA DEALER',completa)
            if diagnostico:
                diag=dealer.copy()
                for c,razon in rechazados:
                    cv2.drawContours(diag,[c],-1,(0,165,255),1)
                    x,y,_,_=cv2.boundingRect(c)
                    cv2.putText(diag,razon,(max(0,x),max(12,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,165,255),1)
                cv2.imshow('ESTABLE - RECHAZADOS',diag)
                for nombre,mascara in mascaras.items(): cv2.imshow('ESTABLE - '+nombre.upper(),mascara)
            ultimo=(vista,completa,corregidas,indices,mascaras)
            tecla=cv2.waitKey(25)&0xff
            if tecla==ord('q'):break
            if tecla==ord('r'): historial.clear();print('Medicion reiniciada.')
            if tecla==ord('d'):
                diagnostico=not diagnostico
                if not diagnostico:
                    for nombre in ('RECHAZADOS','OTSU','RESPALDO','CANNY'):
                        cv2.destroyWindow('ESTABLE - '+nombre)
            if tecla==ord('s'):
                carpeta=original.ROOT/'debug'/'detector_estable'
                carpeta.mkdir(parents=True,exist_ok=True)
                fecha=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                cv2.imwrite(str(carpeta/f'{fecha}_contornos.png'),vista)
                cv2.imwrite(str(carpeta/f'{fecha}_camara.png'),completa)
                for nombre,mascara in mascaras.items():cv2.imwrite(str(carpeta/f'{fecha}_{nombre}.png'),mascara)
                for i,(c,idx) in enumerate(zip(corregidas,indices),1):
                    cv2.imwrite(str(carpeta/f'{fecha}_carta_{i}.png'),c)
                    cv2.imwrite(str(carpeta/f'{fecha}_indice27_{i}.png'),idx)
                print(f'Guardado: {carpeta} | 4 cartas: {estabilidad:.1f}% en {len(historial)} fotogramas')
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__=='__main__':main()
