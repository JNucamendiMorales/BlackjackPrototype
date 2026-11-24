"""Detector experimental de cartas individuales para jugadores.

No modifica ni depende de las funciones de separación del dealer.
"""

import cv2
import numpy as np

MIN_AREA = 3000
CARD_RATIO = 250 / 350


def ordenar_puntos(puntos):
    """Orden circular de los cuatro vértices."""
    puntos = np.asarray(puntos, dtype=np.float32).reshape(4, 2)

    centro = puntos.mean(axis=0)

    angulos = np.arctan2(
        puntos[:, 1] - centro[1],
        puntos[:, 0] - centro[0]
    )

    puntos = puntos[np.argsort(angulos)]

    inicio = np.argmin(puntos.sum(axis=1))
    puntos = np.roll(puntos, -int(inicio), axis=0)

    return puntos


def detectar_candidatos(imagen, binaria):
    """Obtiene candidatos de cartas completas."""
    contornos, _ = cv2.findContours(
        binaria,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidatos = []

    for contorno in contornos:
        area = cv2.contourArea(contorno)

        if area < MIN_AREA:
            continue

        rectangulo = cv2.minAreaRect(contorno)
        ancho, alto = rectangulo[1]

        if min(ancho, alto) <= 0:
            continue

        proporcion = min(ancho, alto) / max(ancho, alto)

        if not 0.55 <= proporcion <= 0.90:
            continue

        area_rectangulo = ancho * alto
        relleno = area / area_rectangulo

        if relleno < 0.70:
            continue

        perimetro = cv2.arcLength(contorno, True)

        aproximacion = cv2.approxPolyDP(
            cv2.convexHull(contorno),
            0.025 * perimetro,
            True
        )

        if len(aproximacion) == 4:
            puntos = ordenar_puntos(aproximacion)
            metodo = "4 vertices"
        else:
            puntos = ordenar_puntos(
                cv2.boxPoints(rectangulo)
            )
            metodo = "rectangulo rotado"

        candidatos.append((puntos, metodo))

    return candidatos


def calcular_iou(puntos_a, puntos_b):
    """Calcula la superposición entre dos polígonos."""
    a = ordenar_puntos(puntos_a)
    b = ordenar_puntos(puntos_b)

    area_a = abs(cv2.contourArea(a))
    area_b = abs(cv2.contourArea(b))

    if area_a <= 0 or area_b <= 0:
        return 0.0

    interseccion, _ = cv2.intersectConvexConvex(a, b)

    union = area_a + area_b - interseccion

    return interseccion / union if union > 0 else 0.0


def detectar_cartas(imagen):
    """Detecta cartas completas utilizando dos umbrales."""
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    suavizada = cv2.GaussianBlur(gris, (5, 5), 0)

    _, binaria_otsu = cv2.threshold(
        suavizada,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    _, binaria_fija = cv2.threshold(
        suavizada,
        135,
        255,
        cv2.THRESH_BINARY
    )

    candidatos = detectar_candidatos(imagen, binaria_otsu)
    candidatos += detectar_candidatos(imagen, binaria_fija)

    candidatos.sort(
        key=lambda elemento: (
            0 if elemento[1] == "4 vertices" else 1
        )
    )

    resultados = []

    for puntos, metodo in candidatos:
        duplicado = any(
            calcular_iou(puntos, existente) >= 0.45
            for existente, _ in resultados
        )

        if not duplicado:
            resultados.append((puntos, metodo))

    resultados.sort(
        key=lambda elemento: elemento[0][:, 0].mean()
    )

    return resultados
