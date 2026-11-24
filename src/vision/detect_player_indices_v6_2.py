"""
BlackjackPrototype - PLAYER V6.2

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

ANGLE_TOLERANCE = 25.0
MIN_EDGE_LENGTH = 8.0

MIN_HORIZONTAL_STEP = 5
MIN_VERTICAL_STEP = 5

INDEX_MARGIN_X = -1
INDEX_MARGIN_Y = -2

INDEX_WIDTH_FACTOR = 0.60
INDEX_HEIGHT_FACTOR = 1.00

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

        derecha = False
        abajo = False

        for p in vecinos:

            dx = float(p[0] - actual[0])
            dy = float(p[1] - actual[1])

            longitud = np.hypot(dx, dy)

            if longitud < MIN_EDGE_LENGTH:
                continue

            angulo = np.degrees(
                np.arctan2(dy, dx)
            )

            if dx > 0 and abs(angulo) <= ANGLE_TOLERANCE:
                derecha = True

            if dy > 0 and abs(angulo - 90) <= ANGLE_TOLERANCE:
                abajo = True

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



def obtener_direcciones(contorno, esquina):
    """Obtiene las direcciones locales de los bordes."""

    perimetro = cv2.arcLength(contorno, True)

    aproximacion = cv2.approxPolyDP(
        contorno,
        CORNER_EPSILON * perimetro,
        True
    )

    puntos = aproximacion.reshape(-1, 2).astype(np.float32)

    origen = np.array(esquina, dtype=np.float32)

    i = int(np.argmin(
        np.linalg.norm(puntos - origen, axis=1)
    ))

    vecinos = [
        puntos[(i - 1) % len(puntos)],
        puntos[(i + 1) % len(puntos)]
    ]

    derecha = None
    abajo = None

    for vecino in vecinos:

        vector = vecino - origen
        longitud = np.linalg.norm(vector)

        if longitud < MIN_EDGE_LENGTH:
            continue

        dx, dy = vector

        angulo = np.degrees(
            np.arctan2(dy, dx)
        )

        if dx > 0 and abs(angulo) <= ANGLE_TOLERANCE:
            derecha = vector / longitud

        if dy > 0 and abs(angulo - 90) <= ANGLE_TOLERANCE:
            abajo = vector / longitud

    return derecha, abajo


def recortar_indice_orientado(
    imagen, contorno, esquina, ancho, alto
):
    """Construye y rectifica el cuadrilatero del indice."""

    derecha, abajo = obtener_direcciones(
        contorno, esquina
    )

    if derecha is None or abajo is None:
        return None, None

    origen = np.array(esquina, dtype=np.float32)

    p1 = (
        origen
        + derecha * INDEX_MARGIN_X
        + abajo * INDEX_MARGIN_Y
    )

    p2 = p1 + derecha * (ancho - 1)
    p3 = p2 + abajo * (alto - 1)
    p4 = p1 + abajo * (alto - 1)

    vertices = np.array(
        [p1, p2, p3, p4],
        dtype=np.float32
    )

    destino = np.array([
        [0, 0],
        [ancho - 1, 0],
        [ancho - 1, alto - 1],
        [0, alto - 1]
    ], dtype=np.float32)

    matriz = cv2.getPerspectiveTransform(
        vertices, destino
    )

    recorte = cv2.warpPerspective(
        imagen,
        matriz,
        (ancho, alto),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    return recorte, vertices


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
        16,
        int(ancho_escalon * INDEX_WIDTH_FACTOR)
    )

    alto_indice = max(
        20,
        int(ancho_escalon * INDEX_HEIGHT_FACTOR)
    ) + 2

    # PLAYER V5:
    # Orden de reparto desde la perspectiva de la camara:
    # derecha-arriba hacia izquierda-abajo.
    # Se invierte despues de calcular las dimensiones.
    esquinas = list(reversed(esquinas))

    resultados = []

    alto_imagen, ancho_imagen = imagen.shape[:2]

    for numero, (px, py) in enumerate(
        esquinas,
        start=1
    ):

        recorte, vertices = recortar_indice_orientado(
            imagen,
            contorno,
            (px, py),
            ancho_indice,
            alto_indice
        )

        if recorte is None:
            continue

        x, y, w, h = cv2.boundingRect(vertices)

        resultados.append({
            "numero": numero,
            "esquina": (px, py),
            "rectangulo": (x, y, x + w, y + h),
            "vertices": vertices.tolist(),
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

        vertices = np.rint(
            np.array(indice["vertices"])
        ).astype(np.int32)

        cv2.polylines(
            vista,
            [vertices],
            True,
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
