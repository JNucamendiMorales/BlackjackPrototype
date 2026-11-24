"""Detector V2 para cartas individuales de jugadores."""

import cv2
import numpy as np

MIN_AREA = 3000
MIN_RATIO = 0.48
MAX_RATIO = 1.0


def ordenar(puntos):
    """Orden circular de cuatro puntos."""
    puntos = np.asarray(puntos, dtype=np.float32).reshape(4, 2)
    centro = puntos.mean(axis=0)

    angulos = np.arctan2(
        puntos[:, 1] - centro[1],
        puntos[:, 0] - centro[0]
    )

    puntos = puntos[np.argsort(angulos)]
    inicio = np.argmin(puntos.sum(axis=1))

    return np.roll(puntos, -int(inicio), axis=0)


def proporcion_vertices(puntos):
    """Calcula proporción usando los lados del cuadrilátero."""
    p = ordenar(puntos)

    lados = [
        np.linalg.norm(p[(i + 1) % 4] - p[i])
        for i in range(4)
    ]

    ancho = (lados[0] + lados[2]) / 2
    alto = (lados[1] + lados[3]) / 2

    if min(ancho, alto) <= 0:
        return 0

    return min(ancho, alto) / max(ancho, alto)


def candidatos(imagen, binaria):
    contornos, _ = cv2.findContours(
        binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    encontrados = []

    for contorno in contornos:
        area = cv2.contourArea(contorno)

        if area < MIN_AREA:
            continue

        casco = cv2.convexHull(contorno)
        perimetro = cv2.arcLength(casco, True)

        aproximacion = cv2.approxPolyDP(
            casco, 0.02 * perimetro, True
        )

        if len(aproximacion) == 4:
            puntos = ordenar(aproximacion)
            metodo = "4 vertices"
        else:
            rect = cv2.minAreaRect(contorno)
            puntos = ordenar(cv2.boxPoints(rect))
            metodo = "rectangulo"

        proporcion = proporcion_vertices(puntos)

        if not MIN_RATIO <= proporcion <= MAX_RATIO:
            continue

        area_poligono = abs(cv2.contourArea(puntos))

        if area_poligono <= 0:
            continue

        relleno = area / area_poligono

        if relleno < 0.65:
            continue

        encontrados.append((puntos, metodo))

    return encontrados


def iou(a, b):
    a = ordenar(a)
    b = ordenar(b)

    area_a = abs(cv2.contourArea(a))
    area_b = abs(cv2.contourArea(b))

    interseccion, _ = cv2.intersectConvexConvex(a, b)

    union = area_a + area_b - interseccion

    return interseccion / union if union > 0 else 0


def detectar_cartas(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)

    _, otsu = cv2.threshold(
        gris, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    _, fija = cv2.threshold(
        gris, 135, 255, cv2.THRESH_BINARY
    )

    encontrados = candidatos(imagen, otsu)
    encontrados += candidatos(imagen, fija)

    encontrados.sort(
        key=lambda elemento: (
            0 if elemento[1] == "4 vertices" else 1
        )
    )

    resultados = []

    for puntos, metodo in encontrados:
        if any(iou(puntos, existente) >= 0.45
               for existente, _ in resultados):
            continue

        resultados.append((puntos, metodo))

    resultados.sort(
        key=lambda elemento: elemento[0][:, 0].mean()
    )

    return resultados
