"""Diagnóstico de contornos del dealer, basado en el detector V2.
Ejecutar: python src/vision/detect_dealer_cards_diagnostico.py
Q salir | D binarios | S guardar captura y reporte | R reiniciar estadísticas
"""
from pathlib import Path
from datetime import datetime
import json
import cv2
import numpy as np
import detect_dealer_cards as original

W, H = original.CARD_WIDTH, original.CARD_HEIGHT


def ordenar(puntos):
    pts = np.asarray(puntos, dtype=np.float32).reshape(4, 2)
    centro = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - centro[1], pts[:, 0] - centro[0])
    pts = pts[np.argsort(ang)]
    pts = np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)
    v1, v2 = pts[1] - pts[0], pts[2] - pts[1]
    if v1[0] * v2[1] - v1[1] * v2[0] < 0:
        pts = pts[[0, 3, 2, 1]]
    return pts.astype(np.float32)


def cuadrilatero(contorno):
    casco = cv2.convexHull(contorno)
    perimetro = cv2.arcLength(casco, True)
    for factor in (0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
        aprox = cv2.approxPolyDP(casco, factor * perimetro, True)
        if len(aprox) == 4 and cv2.isContourConvex(aprox):
            pts = ordenar(aprox)
            if min(np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)) > 12:
                return pts, '4 vertices'
    return ordenar(cv2.boxPoints(cv2.minAreaRect(contorno))), 'rectangulo'


def analizar(imagen, binaria, fuente):
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aceptados, candidatos = [], []
    area_max = imagen.shape[0] * imagen.shape[1] * .95
    for numero, c in enumerate(contornos, 1):
        area = cv2.contourArea(c)
        if area < max(100, original.MIN_CARD_AREA * .15):
            continue  # Ruido pequeño, no merece anotación.
        rect = cv2.minAreaRect(c)
        a, b = rect[1]
        razon = max(a, b) / min(a, b) if min(a, b) > 0 else 0
        relleno = area / (a * b) if a * b > 0 else 0
        info = {
            'fuente': fuente, 'id': numero, 'area': round(area, 1),
            'razon': round(razon, 3), 'relleno': round(relleno, 3),
            'centro': [round(float(v), 1) for v in rect[0]],
            'estado': '', 'motivo': '',
        }
        if not original.MIN_CARD_AREA <= area <= area_max:
            info['estado'] = 'rechazado'
            info['motivo'] = 'area'
        elif min(a, b) <= 0:
            info['estado'] = 'rechazado'
            info['motivo'] = 'dimensiones nulas'
        elif 1.05 <= razon <= 2.30 and relleno >= .68:
            pts, metodo = cuadrilatero(c)
            aceptados.append((pts, metodo, fuente))
            info['estado'] = 'aceptado'
            info['motivo'] = metodo
        else:
            # Mantiene EXACTAMENTE la lógica de aceptación de V2.
            try:
                separados = original.split_connected_contour(c, binaria)
                for pts in separados:
                    aceptados.append((ordenar(pts), 'separacion', fuente))
                info['estado'] = 'aceptado' if separados else 'rechazado'
                info['motivo'] = f'separacion: {len(separados)}' if separados else (
                    'razon fuera de rango' if not 1.05 <= razon <= 2.30 else 'relleno < 0.68'
                )
            except Exception as exc:
                info['estado'] = 'rechazado'
                info['motivo'] = f'error separacion: {str(exc)[:55]}'
        candidatos.append((c, info))
    return aceptados, candidatos


def combinar(principales, secundarios):
    # Igual criterio de duplicados de la V2, pero sin truncar antes de combinar.
    resultados = list(principales)
    for pts, metodo, fuente in secundarios:
        centro = pts.mean(axis=0)
        ancho = max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3]))
        alto = max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1]))
        repetido = any(np.linalg.norm(centro - otro.mean(axis=0)) < .35 * min(ancho, alto)
                       for otro, _, _ in resultados)
        if not repetido:
            resultados.append((pts, metodo + ' respaldo', fuente))
    resultados.sort(key=lambda item: item[0][:, 0].mean())
    return resultados[:original.MAX_CARDS]


def enderezar(imagen, pts):
    mx, my = W * .025, H * .025
    destino = np.array([[mx, my], [W-1-mx, my], [W-1-mx, H-1-my], [mx, H-1-my]], np.float32)
    return cv2.warpPerspective(imagen, cv2.getPerspectiveTransform(ordenar(pts), destino), (W, H))


def mosaico(imagenes):
    if not imagenes:
        return np.zeros((260, 180, 3), np.uint8)
    bloques = []
    for i, img in enumerate(imagenes, 1):
        fondo = np.full((270, 190, 3), 45, np.uint8)
        h, w = img.shape[:2]
        escala = min(178 / w, 234 / h)
        mini = cv2.resize(img, (max(1, int(w * escala)), max(1, int(h * escala))))
        fondo[29:29+mini.shape[0], 6:6+mini.shape[1]] = mini
        cv2.putText(fondo, f'Indice {i}', (7, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
        bloques.append(fondo)
    return np.hstack(bloques)


def dibujar_diagnostico(imagen, candidatos, aceptados):
    salida = imagen.copy()
    # Primero candidatos, luego aceptados para que los contornos verdes destaquen.
    for c, info in candidatos:
        color = (0, 200, 255) if info['estado'] == 'rechazado' else (100, 100, 100)
        cv2.drawContours(salida, [c], -1, color, 2 if info['estado'] == 'rechazado' else 1)
        if info['estado'] == 'rechazado':
            x, y, _, _ = cv2.boundingRect(c)
            texto = f"{info['fuente'][0]}:{info['motivo']}"
            cv2.putText(salida, texto[:32], (x, max(15, y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .37, color, 1)
    for i, (pts, metodo, fuente) in enumerate(aceptados, 1):
        cv2.polylines(salida, [pts.astype(np.int32)], True, (0, 255, 0), 2)
        for p in pts:
            cv2.circle(salida, tuple(p.astype(int)), 3, (0, 0, 255), -1)
        x, y = pts[0].astype(int)
        cv2.putText(salida, f'{i}:{metodo}', (max(0, x), max(18, y-7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
    return salida


def main():
    cfg = original.cam_cfg
    cam = cv2.VideoCapture(cfg['index'])
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, cfg['width'])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg['height'])
    if not cam.isOpened():
        raise SystemExit('No se pudo abrir la camara.')
    diagnostico = False
    cuadros = completos = 0
    print('DIAGNOSTICO V2 | Q salir | D mostrar binarios | S guardar reporte | R reiniciar')
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = map(int, original.zones['DEALER'])
            x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
            y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
            if x2 <= x1 or y2 <= y1:
                raise ValueError('Zona DEALER invalida.')
            dealer = frame[y1:y2, x1:x2].copy()
            _, binaria = original.make_binary(dealer)
            gris = cv2.cvtColor(dealer, cv2.COLOR_BGR2GRAY)
            _, respaldo = cv2.threshold(gris, 135, 255, cv2.THRESH_BINARY)
            respaldo = cv2.morphologyEx(respaldo, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            p, cp = analizar(dealer, binaria, 'original')
            s, cs = analizar(dealer, respaldo, 'respaldo')
            cartas = combinar(p, s)
            cuadros += 1
            completos += len(cartas) == 4
            vista = dibujar_diagnostico(dealer, cp + cs, cartas)
            cv2.putText(vista, f'Cartas {len(cartas)} | 4/4: {100*completos/cuadros:.1f}% | S guardar',
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 255, 255), 2)
            indices = []
            for pts, _, _ in cartas:
                carta = enderezar(dealer, pts)
                indices.append(carta[:int(H * .49), :int(W * .27)])
            cv2.imshow('DIAGNOSTICO - CONTORNOS (amarillo = rechazado)', vista)
            cv2.imshow('DIAGNOSTICO - INDICES 27%', mosaico(indices))
            if diagnostico:
                cv2.imshow('DIAGNOSTICO - BINARIO ORIGINAL', binaria)
                cv2.imshow('DIAGNOSTICO - BINARIO RESPALDO', respaldo)
            tecla = cv2.waitKey(25) & 0xFF
            if tecla == ord('q'):
                break
            if tecla == ord('r'):
                cuadros = completos = 0
            if tecla == ord('d'):
                diagnostico = not diagnostico
                if not diagnostico:
                    cv2.destroyWindow('DIAGNOSTICO - BINARIO ORIGINAL')
                    cv2.destroyWindow('DIAGNOSTICO - BINARIO RESPALDO')
            if tecla == ord('s'):
                carpeta = original.ROOT / 'debug' / 'diagnostico_v2'
                carpeta.mkdir(parents=True, exist_ok=True)
                fecha = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                for nombre, img in [('contornos', vista), ('dealer', dealer),
                                    ('binario_original', binaria), ('binario_respaldo', respaldo)]:
                    cv2.imwrite(str(carpeta / f'{fecha}_{nombre}.png'), img)
                reporte = {
                    'fecha': fecha, 'cartas_detectadas': len(cartas),
                    'estabilidad_4_cartas': round(completos / cuadros, 4),
                    'area_minima_configurada': original.MIN_CARD_AREA,
                    'candidatos': [info for _, info in cp + cs],
                }
                ruta = carpeta / f'{fecha}_reporte.json'
                ruta.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding='utf-8')
                print(f'Guardado: {ruta}')
                for info in reporte['candidatos']:
                    print(f"  {info['fuente']:8} area={info['area']:8.0f} razon={info['razon']:.2f} "
                          f"relleno={info['relleno']:.2f} {info['estado']:9} {info['motivo']}")
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
