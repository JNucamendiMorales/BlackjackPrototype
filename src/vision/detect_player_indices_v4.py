"""
BlackjackPrototype - PLAYER V4

Detector reutilizable de indices superiores izquierdos
para cartas superpuestas en escalera diagonal.

Basado en PLAYER V3.8.

No abre la camara.
No modifica archivos.
No utiliza la CNN.
"""

import cv2
import numpy as np


MIN_CONTOUR_AREA = 3000

CORNER_EPSILON = 0.012

MIN_HORIZONTAL_STEP = 5
MIN_VERTICAL_STEP = 5

INDEX_MARGIN_X = 1
INDEX_MARGIN_Y = -1

INDEX_WIDTH_FACTOR = 0.48
INDEX_HEIGHT_FACTOR = 0.90

DEFAULT_STEP_WIDTH = 35.0


def detectar_esquinas(imagen):
    """Devuelve esquinas superiores izquierdas y contorno exterior."""

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    gris = cv2.GaussianBlur(
        gris,
        (5, 5),
        0
    )

    _, mascara = cv2.threshold(
        gris,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos:
        return [], None

    contorno = max(
        contornos,
        key=cv2.contourArea
    )

    if cv2.contourArea(contorno) < MIN_CONTOUR_AREA:
        return [], None

    perimetro = cv2.arcLength(
        contorno,
        True
    )

    aproximacion = cv2.approxPolyDP(
        contorno,
        CORNER_EPSILON * perimetro,
        True
    )

    puntos = aproximacion.reshape(-1, 2)

    candidatos = []

    n = len(puntos)

    for i in range(n):

        actual = puntos[i]

        anterior = puntos[(i - 1) % n]
        siguiente = puntos[(i + 1) % n]

        vecinos = [anterior, siguiente]

        derecha = any(
            p[0] > actual[0] + 8
            and abs(p[1] - actual[1]) < 12
            for p in vecinos
        )

        abajo = any(
            p[1] > actual[1] + 8
            and abs(p[0] - actual[0]) < 12
            for p in vecinos
        )

        if derecha and abajo:
            candidatos.append(
                tuple(map(int, actual))
            )

    candidatos.sort(
        key=lambda p: p[0]
    )

    filtrados = []

    for punto in candidatos:

        if not filtrados:
            filtrados.append(punto)
            continue

        ultimo = filtrados[-1]

        dx = punto[0] - ultimo[0]
        dy = punto[1] - ultimo[1]

        if (
            dx > MIN_HORIZONTAL_STEP
            and dy < -MIN_VERTICAL_STEP
        ):
            filtrados.append(punto)

    return filtrados, contorno


def extraer_indices(imagen):
    """
    Extrae indices de una imagen PLAYER_1.

    Devuelve un diccionario:
        esquinas
        contorno
        indices

    Cada elemento de indices contiene:
        numero
        esquina
        rectangulo
        imagen
    """

    esquinas, contorno = detectar_esquinas(imagen)

    separaciones = [
        esquinas[j + 1][0] - esquinas[j][0]
        for j in range(len(esquinas) - 1)
        if esquinas[j + 1][0] > esquinas[j][0]
    ]

    if separaciones:
        ancho_escalon = float(
            np.median(separaciones)
        )
    else:
        ancho_escalon = DEFAULT_STEP_WIDTH

    ancho_indice = max(
        12,
        int(ancho_escalon * INDEX_WIDTH_FACTOR)
    )

    alto_indice = max(
        15,
        int(ancho_escalon * INDEX_HEIGHT_FACTOR)
    ) + 2

    resultados = []

    alto_imagen, ancho_imagen = imagen.shape[:2]

    for numero, (px, py) in enumerate(
        esquinas,
        start=1
    ):

        rx1 = max(
            0,
            px + INDEX_MARGIN_X
        )

        ry1 = max(
            0,
            py + INDEX_MARGIN_Y
        )

        rx2 = min(
            ancho_imagen,
            rx1 + ancho_indice
        )

        ry2 = min(
            alto_imagen,
            ry1 + alto_indice
        )

        recorte = imagen[
            ry1:ry2,
            rx1:rx2
        ].copy()

        if recorte.size == 0:
            continue

        resultados.append({
            "numero": numero,
            "esquina": (px, py),
            "rectangulo": (rx1, ry1, rx2, ry2),
            "imagen": recorte
        })

    return {
        "esquinas": esquinas,
        "contorno": contorno,
        "indices": resultados
    }


def dibujar_resultados(imagen, resultados):
    """Genera una vista previa sin alterar la imagen original."""

    vista = imagen.copy()

    contorno = resultados["contorno"]

    if contorno is not None:
        cv2.drawContours(
            vista,
            [contorno],
            -1,
            (0, 255, 0),
            2
        )

    for indice in resultados["indices"]:

        numero = indice["numero"]
        px, py = indice["esquina"]

        rx1, ry1, rx2, ry2 = indice["rectangulo"]

        cv2.rectangle(
            vista,
            (rx1, ry1),
            (rx2, ry2),
            (255, 0, 0),
            2
        )

        cv2.circle(
            vista,
            (px, py),
            6,
            (255, 0, 255),
            -1
        )

        cv2.putText(
            vista,
            str(numero),
            (px + 7, py - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    cv2.putText(
        vista,
        f"Indices: {len(resultados['indices'])}",
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        2
    )

    return vista
