"""Diagnostico OpenCV V2: contornos y comparacion de indices, sin PyTorch.
Ejecutar: python src/vision/detect_dealer_cards_v2.py
"""
import cv2
import numpy as np
import detect_dealer_cards as original
from pathlib import Path
from datetime import datetime

W, H = original.CARD_WIDTH, original.CARD_HEIGHT


def ordenar(puntos):
    pts = np.asarray(puntos, dtype=np.float32).reshape(4, 2)
    centro = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - centro[1], pts[:, 0] - centro[0])
    pts = pts[np.argsort(ang)]
    pts = np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[1]
    cruz_2d = v1[0] * v2[1] - v1[1] * v2[0]
    if cruz_2d < 0:
        pts = pts[[0, 3, 2, 1]]
    return pts.astype(np.float32)


def cuadrilatero(contorno):
    casco = cv2.convexHull(contorno)
    perimetro = cv2.arcLength(casco, True)
    for factor in (0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
        aprox = cv2.approxPolyDP(casco, factor * perimetro, True)
        if len(aprox) == 4 and cv2.isContourConvex(aprox):
            pts = ordenar(aprox)
            if min(np.linalg.norm(pts[(i+1) % 4] - pts[i]) for i in range(4)) > 12:
                return pts, '4 vertices'
    return ordenar(cv2.boxPoints(cv2.minAreaRect(contorno))), 'rectangulo'


def detectar(imagen, binaria):
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    encontrados = []
    for c in contornos:
        area = cv2.contourArea(c)
        if not original.MIN_CARD_AREA <= area <= imagen.shape[0] * imagen.shape[1] * 0.95:
            continue
        rect = cv2.minAreaRect(c)
        a, b = rect[1]
        if min(a, b) <= 0:
            continue
        razon = max(a, b) / min(a, b)
        relleno = area / (a * b)
        x, y, ancho, alto = cv2.boundingRect(c)
        estimadas = ancho / (alto * original.CARD_RATIO)

        if estimadas >= 1.65:
            print(
                f"[POSIBLE GRUPO] "
                f"Ancho={ancho} | "
                f"Alto={alto} | "
                f"Estimadas={estimadas:.2f} | "
                f"Ratio={razon:.2f} | "
                f"Relleno={relleno:.2f}",
                flush=True
            )

        print(
            f"[DIAGNOSTICO] "
            f"Area={area:.0f} | "
            f"Ancho={ancho} | "
            f"Alto={alto} | "
            f"Ratio={razon:.2f} | "
            f"Relleno={relleno:.2f} | "
            f"Cartas estimadas={estimadas:.2f}",
            flush=True
        )


        # Detectar primero posibles grupos de cartas verticales.
        # Usamos un umbral conservador para evitar dividir cartas inclinadas.
        posibles_cartas = ancho / (alto * original.CARD_RATIO)

        if posibles_cartas >= 1.45:
            separadas = original.split_connected_contour(c, binaria)

            if len(separadas) >= 2:
                for pts in separadas:
                    encontrados.append((ordenar(pts), 'separacion'))
                continue

        # Si no se pudo separar, evaluar como carta individual.
        if 1.00 <= razon <= 2.30 and relleno >= 0.68:
            encontrados.append(cuadrilatero(c))
        else:
            for pts in original.split_connected_contour(c, binaria):
                encontrados.append((ordenar(pts), 'separacion'))
    encontrados.sort(key=lambda elemento: elemento[0][:, 0].mean())
    return encontrados



def superposicion(pts_a, pts_b):
    """Calcula la intersección sobre unión (IoU) de dos polígonos."""
    a = ordenar(pts_a).astype(np.float32)
    b = ordenar(pts_b).astype(np.float32)

    area_a = abs(cv2.contourArea(a))
    area_b = abs(cv2.contourArea(b))

    if area_a <= 0 or area_b <= 0:
        return 0.0

    area_interseccion, _ = cv2.intersectConvexConvex(a, b)
    area_union = area_a + area_b - area_interseccion

    return area_interseccion / area_union if area_union > 0 else 0.0


def detectar_reforzado(imagen, binaria):
    """Combina ambos umbrales y elimina detecciones duplicadas."""
    principales = detectar(imagen, binaria)

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, alternativa = cv2.threshold(
        gris, 135, 255, cv2.THRESH_BINARY
    )
    alternativa = cv2.morphologyEx(
        alternativa,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8)
    )

    secundarios = detectar(imagen, alternativa)

    candidatos = list(principales)
    candidatos.extend(
        (pts, metodo + " respaldo")
        for pts, metodo in secundarios
    )

    # Preferir detecciones individuales con cuatro vértices.
    def prioridad(elemento):
        pts, metodo = elemento
        if metodo.startswith("4 vertices"):
            return 0
        if metodo.startswith("rectangulo"):
            return 1
        return 2

    candidatos.sort(key=prioridad)

    resultados = []

    for pts, metodo in candidatos:
        duplicado = False

        for existente, _ in resultados:
            if superposicion(pts, existente) >= 0.45:
                duplicado = True
                break

        if not duplicado:
            resultados.append((pts, metodo))

    resultados.sort(
        key=lambda elemento: elemento[0][:, 0].mean()
    )

    return resultados[:original.MAX_CARDS], alternativa


def enderezar(imagen, pts, margen=0.025):
    # Un margen en la imagen corregida conserva algo de fondo alrededor.
    mx, my = W * margen, H * margen
    destino = np.array([[mx, my], [W-1-mx, my], [W-1-mx, H-1-my], [mx, H-1-my]], np.float32)
    matriz = cv2.getPerspectiveTransform(ordenar(pts), destino)
    return cv2.warpPerspective(imagen, matriz, (W, H))


def indice(carta, fraccion_ancho, fraccion_alto):
    alto, ancho = carta.shape[:2]
    return carta[:int(alto * fraccion_alto), :int(ancho * fraccion_ancho)]


def mosaico(imagenes, titulo):
    if not imagenes:
        return np.zeros((270, 190, 3), np.uint8)
    bloques = []
    for numero, img in enumerate(imagenes, 1):
        fondo = np.full((270, 190, 3), 45, np.uint8)
        h, w = img.shape[:2]
        escala = min(178/w, 234/h)
        mini = cv2.resize(img, (max(1, int(w*escala)), max(1, int(h*escala))))
        fondo[29:29+mini.shape[0], 6:6+mini.shape[1]] = mini
        cv2.putText(fondo, f'{titulo} {numero}', (7, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
        bloques.append(fondo)
    return np.hstack(bloques)


def main():
    config = original.cam_cfg
    camara = cv2.VideoCapture(config['index'])
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
    if not camara.isOpened():
        raise SystemExit('No se pudo abrir la camara. Cierra otros programas que la usen.')
    print('DETECTOR V2 | Q: salir | D: diagnostico | S: guardar imagenes')
    print('Exposicion:', camara.get(cv2.CAP_PROP_EXPOSURE),
          '| Ganancia:', camara.get(cv2.CAP_PROP_GAIN),
          '| Brillo:', camara.get(cv2.CAP_PROP_BRIGHTNESS),
          '| Autoexposicion:', camara.get(cv2.CAP_PROP_AUTO_EXPOSURE))
    debug = False
    try:
        while True:
            ok, frame = camara.read()
            if not ok:
                print('No se pudo leer la camara.')
                break
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = map(int, original.zones['DEALER'])
            x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
            y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
            if x2 <= x1 or y2 <= y1:
                raise ValueError('Zona DEALER invalida en config/game_zones.json')
            dealer = frame[y1:y2, x1:x2].copy()
            _, binaria = original.make_binary(dealer)
            cartas, binaria_respaldo = detectar_reforzado(dealer, binaria)
            pantalla = dealer.copy()
            corregidas, indices27, indices38 = [], [], []
            for i, (pts, metodo) in enumerate(cartas, 1):
                cv2.polylines(pantalla, [pts.astype(np.int32)], True, (0,255,0), 2)
                for punto in pts:
                    cv2.circle(pantalla, tuple(punto.astype(int)), 4, (0,0,255), -1)
                px, py = pts[0].astype(int)
                cv2.putText(pantalla, f'{i}: {metodo}', (max(0,px), max(18,py-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (0,255,255), 1)
                carta = enderezar(dealer, pts)
                corregidas.append(carta)
                indices27.append(indice(carta, .27, .49))
                indices38.append(indice(carta, .38, .50))
            cv2.putText(pantalla, f'Cartas: {len(cartas)} | Q salir | S guardar',
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,255), 2)
            # Mostrar en la imagen COMPLETA el rectangulo de la zona DEALER.
            vista_completa = frame.copy()
            cv2.rectangle(vista_completa, (x1, y1), (x2-1, y2-1), (255, 255, 0), 2)
            cv2.putText(vista_completa, 'ZONA DEALER (celeste)',
                        (max(0, x1), max(20, y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 0), 2)
            cv2.imshow('V2 - CAMARA COMPLETA Y ZONA DEALER', vista_completa)
            cv2.imshow('V2 - CONTORNOS Y ESQUINAS', pantalla)
            cv2.imshow('V2 - CARTAS CORREGIDAS', mosaico(corregidas, 'Carta'))
            cv2.imshow('V2 - INDICE ORIGINAL 27%', mosaico(indices27, '27%'))
            cv2.imshow('V2 - INDICE AMPLIADO 38%', mosaico(indices38, '38%'))
            if debug:
                cv2.imshow('V2 - BINARIO', binaria)
                cv2.imshow('V2 - BINARIO RESPALDO', binaria_respaldo)
            tecla = cv2.waitKey(25) & 0xFF
            if tecla == ord('q'):
                break
            if tecla == ord('d'):
                debug = not debug
                if not debug:
                    cv2.destroyWindow('V2 - BINARIO')
                    cv2.destroyWindow('V2 - BINARIO RESPALDO')
            if tecla == ord('s'):
                carpeta = original.ROOT / 'debug' / 'detector_v2'
                carpeta.mkdir(parents=True, exist_ok=True)
                fecha = datetime.now().strftime('%Y%m%d_%H%M%S')
                cv2.imwrite(str(carpeta / f'{fecha}_contornos.png'), pantalla)
                cv2.imwrite(str(carpeta / f'{fecha}_zona_dealer.png'), vista_completa)
                cv2.imwrite(str(carpeta / f'{fecha}_binario_original.png'), binaria)
                cv2.imwrite(str(carpeta / f'{fecha}_binario_respaldo.png'), binaria_respaldo)
                for i, carta in enumerate(corregidas, 1):
                    cv2.imwrite(str(carpeta / f'{fecha}_carta_{i}.png'), carta)
                    cv2.imwrite(str(carpeta / f'{fecha}_indice27_{i}.png'), indices27[i-1])
                    cv2.imwrite(str(carpeta / f'{fecha}_indice38_{i}.png'), indices38[i-1])
                print('Capturas guardadas en:', carpeta)
    finally:
        camara.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
