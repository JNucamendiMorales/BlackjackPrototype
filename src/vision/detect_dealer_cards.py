import cv2
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'config' / 'game_zones.json'
with CONFIG.open(encoding='utf-8') as f:
    cfg = json.load(f)
cam_cfg = cfg['camera']
zones = cfg['zones']

CARD_WIDTH = 250
CARD_HEIGHT = 350

MIN_CARD_AREA = 3000

# Proporción aproximada entre ancho y alto
CARD_RATIO = CARD_WIDTH / CARD_HEIGHT

# Máximo de cartas durante las pruebas
MAX_CARDS = 4

# Tolerancia para detectar regiones
RATIO_TOLERANCE = 0.20

MAX_CARDS = 4


def order_points(points):
    """Orden TL, TR, BR, BL para cartas verticales con inclinacion leve."""
    p = np.asarray(points, dtype=np.float32).reshape(4, 2)
    top = p[np.argsort(p[:, 1])[:2]]
    bottom = p[np.argsort(p[:, 1])[2:]]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def straighten_card(image, corners):
    src = order_points(corners)
    dst = np.array([[0, 0], [CARD_WIDTH - 1, 0],
                    [CARD_WIDTH - 1, CARD_HEIGHT - 1], [0, CARD_HEIGHT - 1]],
                   dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (CARD_WIDTH, CARD_HEIGHT))

# ==========================================
# EXTRAER NUMERO Y PALO DE CADA CARTA
# ==========================================

def extract_card_index(card):

    height, width = card.shape[:2]

    # Esquina superior izquierda
    x1 = 0
    y1 = 0

    # Ajuste del recorte
    x2 = int(width * 0.27)
    y2 = int(height * 0.49)

    index = card[y1:y2, x1:x2]

    return index


def make_binary(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold_value, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
    return threshold_value, binary


def plausible_card(corners, image_shape):

    pts = np.asarray(
        corners,
        dtype=np.float32
    ).reshape(4, 2)

    rectangle = cv2.minAreaRect(pts)

    rw, rh = rectangle[1]

    if rw <= 0 or rh <= 0:
        return False

    width = min(rw, rh)
    height = max(rw, rh)

    if width < 10 or height < 10:
        return False

    ratio = height / width

    area = width * height

    return (
        1.05 <= ratio <= 2.30
        and area >= MIN_CARD_AREA
    )


def split_connected_contour(contour, binary):
    """
    Divide una región que puede contener varias cartas
    verticales colocadas una al lado de otra.

    Utiliza la proporción de las cartas y busca
    separaciones cercanas a las posiciones esperadas.
    """

    x, y, w, h = cv2.boundingRect(contour)

    if h < 30 or w < 20:
        return []

    roi = binary[y:y+h, x:x+w]

    expected_width = h * CARD_RATIO

    estimated_cards = w / expected_width

    number_cards = int(round(estimated_cards))

    number_cards = max(
        1,
        min(MAX_CARDS, number_cards)
    )

    if number_cards < 2:
        return []

    expected_total_width = number_cards * expected_width

    error = abs(w - expected_total_width) / expected_total_width

    if number_cards >= 2:
        print(
            f"[SPLIT] Cartas={number_cards} | "
            f"Ancho real={w} | "
            f"Ancho esperado={expected_total_width:.1f} | "
            f"Error={error:.1%} | "
            f"Tolerancia={RATIO_TOLERANCE:.0%} | "
            f"Resultado={'RECHAZADO' if error > RATIO_TOLERANCE else 'ACEPTADO'}",
            flush=True
        )

    if error > RATIO_TOLERANCE:
        return []

    # Perfil de píxeles blancos por columna
    profile = np.sum(roi > 0, axis=0).astype(np.float32)

    # Suavizar perfil
    kernel_size = 7

    profile = np.convolve(
        profile,
        np.ones(kernel_size) / kernel_size,
        mode="same"
    )

    boundaries = [0]

    # Buscar divisiones entre cartas
    for i in range(1, number_cards):

        expected_cut = int(
            i * w / number_cards
        )

        search_range = int(
            expected_width * 0.15
        )

        left = max(
            boundaries[-1] + 10,
            expected_cut - search_range
        )

        right = min(
            w - 10,
            expected_cut + search_range
        )

        if left >= right:
            return []

        # Buscar el valle más cercano
        local_profile = profile[left:right]

        cut = left + int(
            np.argmin(local_profile)
        )

        boundaries.append(cut)

    boundaries.append(w)

    detected = []

    for left, right in zip(
        boundaries[:-1],
        boundaries[1:]
    ):

        card_width = right - left

        if card_width < expected_width * 0.65:
            return []

        corners = np.array([
            [x + left, y],
            [x + right, y],
            [x + right, y + h],
            [x + left, y + h]
        ], dtype=np.float32)

        detected.append(corners)

    return detected


def detect_cards(image, binary):

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detected = []

    image_area = image.shape[0] * image.shape[1]

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_CARD_AREA:
            continue

        if area > image_area * 0.95:
            continue

        # ==========================================
        # OBTENER RECTANGULO ROTADO
        # ==========================================

        rectangle = cv2.minAreaRect(contour)

        rw, rh = rectangle[1]

        if min(rw, rh) <= 0:
            continue

        corners = cv2.boxPoints(rectangle)

        # ==========================================
        # COMPROBAR GEOMETRIA
        # ==========================================

        width = min(rw, rh)
        height = max(rw, rh)

        ratio = height / width

        fill_ratio = area / (rw * rh)

        # ==========================================
        # DETECTAR CARTA INDIVIDUAL
        # ==========================================

        valid_ratio = 1.00 <= ratio <= 2.30

        valid_area = (
            width * height >= MIN_CARD_AREA
        )

        valid_fill = fill_ratio >= 0.70
        
        print(
        f"[CARTA] "
        f"Area={area:.0f} | "
        f"Ancho={width:.1f} | "
        f"Alto={height:.1f} | "
        f"Ratio={ratio:.2f} | "
        f"Relleno={fill_ratio:.2f} | "
        f"RatioOK={valid_ratio} | "
        f"AreaOK={valid_area} | "
        f"RellenoOK={valid_fill}"
        )

        if valid_ratio and valid_area and valid_fill:

            detected.append(corners)

            continue

        # ==========================================
        # INTENTAR SEPARAR CARTAS UNIDAS
        # ==========================================

        separated = split_connected_contour(
            contour,
            binary
        )

        if separated:

            detected.extend(separated)

    detected = sorted(
        detected,
        key=lambda p: float(np.mean(p[:, 0]))
    )

    return detected[:MAX_CARDS], contours


# ==========================================
# MOSTRAR DIAGNOSTICO
# ==========================================

def show_detection_debug(binary, contours, threshold_value):

    debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(
        debug,
        contours,
        -1,
        (0, 255, 0),
        1
    )

    cv2.putText(
        debug,
        f'Umbral: {threshold_value:.0f} | Contornos: {len(contours)}',
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1
    )

    cv2.imshow('Blackjack - DIAGNOSTICO', debug)


# ==========================================
# INICIAR CAMARA
# ==========================================
def main():
    camera = cv2.VideoCapture(cam_cfg['index'])

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        cam_cfg['width']
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        cam_cfg['height']
    )

    if not camera.isOpened():
        raise SystemExit('[ERROR] No se pudo abrir la camara.')

    print('BLACKJACK - DETECCION DE CARTAS | Q: salir')


    # ==========================================
    # BUCLE PRINCIPAL
    # ==========================================

    try:

        while True:

            ok, frame = camera.read()

            if not ok:
                print('[ERROR] No se pudo leer la camara.')
                break

            h, w = frame.shape[:2]

            # Obtener zona del dealer
            x1, y1, x2, y2 = map(int, zones['DEALER'])

            x1, x2 = sorted((
                max(0, min(w, x1)),
                max(0, min(w, x2))
            ))

            y1, y2 = sorted((
                max(0, min(h, y1)),
                max(0, min(h, y2))
            ))

            if x2 <= x1 or y2 <= y1:
                print('[ERROR] La zona DEALER no es valida.')
                break

            # Recortar zona del dealer
            dealer = frame[y1:y2, x1:x2].copy()

            display = dealer.copy()

            # ==========================================
            # DETECTAR CARTAS
            # ==========================================

            threshold_value, binary = make_binary(dealer)

            cards, contours = detect_cards(dealer, binary)

            show_detection_debug(
                binary,
                contours,
                threshold_value
            )

            # ==========================================
            # PROCESAR CARTAS Y EXTRAER INDICES
            # ==========================================

            corrected_cards = []
            extracted_indices = []

            for i, corners in enumerate(cards):

                ordered = order_points(corners)

                # Dibujar contorno de la carta
                cv2.polylines(
                    display,
                    [ordered.astype(np.int32)],
                    True,
                    (0, 255, 0),
                    2
                )

                # Mostrar numero de carta
                label_x, label_y = ordered[0].astype(int)

                cv2.putText(
                    display,
                    f'CARTA {i+1}',
                    (max(0, label_x), max(15, label_y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                # Corregir perspectiva
                corrected_card = straighten_card(
                    dealer,
                    corners
                )

                corrected_cards.append(corrected_card)

                # Extraer numero y palo
                index = extract_card_index(corrected_card)

                # Ampliar para visualizar mejor
                index_large = cv2.resize(
                    index,
                    (150, 200),
                    interpolation=cv2.INTER_CUBIC
                )

                extracted_indices.append(index_large)

            # ==========================================
            # PREPARAR CARTAS CORREGIDAS
            # ==========================================

            if corrected_cards:

                corrected = np.hstack(corrected_cards)

            else:

                corrected = np.zeros(
                    (CARD_HEIGHT, CARD_WIDTH, 3),
                    dtype=np.uint8
                )

            # Mostrar cantidad de cartas detectadas
            cv2.putText(
                display,
                f'Cartas detectadas: {len(cards)}',
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            # ==========================================
            # PREPARAR INDICES EXTRAIDOS
            # ==========================================

            if extracted_indices:

                indices_display = np.hstack(
                    extracted_indices
                )

            else:

                indices_display = np.zeros(
                    (200, 150, 3),
                    dtype=np.uint8
                )

            # ==========================================
            # MOSTRAR RESULTADOS
            # ==========================================

            cv2.imshow(
                'Blackjack - DEALER',
                display
            )

            cv2.imshow(
                'Blackjack - CARTA CORREGIDA',
                corrected
            )

            cv2.imshow(
                'Blackjack - INDICES EXTRAIDOS',
                indices_display
            )

            # ==========================================
            # SALIR
            # ==========================================

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break


    # ==========================================
    # CERRAR CAMARA
    # ==========================================

    finally:

        camera.release()

        cv2.destroyAllWindows()