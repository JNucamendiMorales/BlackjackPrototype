import cv2
import numpy as np

from pathlib import Path


# ==========================================
# CONFIGURACIÓN
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_IMAGE = PROJECT_ROOT / "dataset" / "raw" / "card_capture.jpg"

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed"
DEBUG_DIR = PROJECT_ROOT / "dataset" / "debug_samples"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# ORDENAR ESQUINAS DE LA CARTA
# ==========================================

def order_card_corners(points):

    points = points.astype(np.float32)

    center = np.mean(points, axis=0)

    angles = np.arctan2(
        points[:, 1] - center[1],
        points[:, 0] - center[0]
    )

    ordered = points[np.argsort(angles)]

    sides = []

    for i in range(4):

        p1 = ordered[i]
        p2 = ordered[(i + 1) % 4]

        length = np.linalg.norm(p2 - p1)

        sides.append(length)

    short_edges = np.argsort(sides)[:2]

    edge_midpoints = []

    for i in short_edges:

        midpoint = (
            ordered[i] + ordered[(i + 1) % 4]
        ) / 2

        edge_midpoints.append((midpoint[1], i))

    top_edge = min(edge_midpoints)[1]

    top_left = ordered[top_edge]

    top_right = ordered[(top_edge + 1) % 4]

    bottom_right = ordered[(top_edge + 2) % 4]

    bottom_left = ordered[(top_edge + 3) % 4]

    return np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dtype=np.float32)


# ==========================================
# CARGAR IMAGEN
# ==========================================

image = cv2.imread(str(INPUT_IMAGE))

if image is None:

    print("[ERROR] No se pudo cargar la imagen.")

    raise SystemExit(1)

print("[INFO] Imagen cargada correctamente.")


# ==========================================
# PREPROCESAMIENTO
# ==========================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

edges = cv2.Canny(
    blurred,
    30,
    100
)


# ==========================================
# DETECCIÓN DE CONTORNOS
# ==========================================

contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

best_contour = None
largest_area = 0


# ==========================================
# BUSCAR POSIBLE CARTA
# ==========================================

for contour in contours:

    area = cv2.contourArea(contour)

    if area < 5000:
        continue

    perimeter = cv2.arcLength(
        contour,
        True
    )

    candidate = None

    for epsilon_factor in [0.015, 0.02, 0.025, 0.03, 0.04]:

        approx = cv2.approxPolyDP(
            contour,
            epsilon_factor * perimeter,
            True
        )

        if len(approx) != 4:
            continue

        if not cv2.isContourConvex(approx):
            continue

        rect = cv2.minAreaRect(contour)

        rect_width, rect_height = rect[1]

        if rect_width == 0 or rect_height == 0:
            continue

        aspect_ratio = max(
            rect_width,
            rect_height
        ) / min(
            rect_width,
            rect_height
        )

        if 1.35 <= aspect_ratio <= 1.85:

            candidate = approx.reshape(4, 2).astype(np.float32)

            break

    # Evaluar el candidato encontrado.
    if candidate is not None:

        if area > largest_area:

            largest_area = area

            best_contour = candidate


# ==========================================
# RECTIFICACIÓN DE PERSPECTIVA
# ==========================================

if best_contour is None:

    print("[ERROR] No se pudo detectar una carta.")

else:

    print("[SUCCESS] Carta detectada correctamente.")

    detected = image.copy()

    cv2.drawContours(
        detected,
        [best_contour.astype(np.int32)],
        -1,
        (0, 255, 0),
        3
    )

    # Ordenar esquinas.
    ordered_points = order_card_corners(
        best_contour
    )

    # Dimensiones de la carta rectificada.
    width = 630
    height = 880

    destination_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # Calcular transformación.
    matrix = cv2.getPerspectiveTransform(
        ordered_points,
        destination_points
    )

    # Aplicar transformación.
    warped = cv2.warpPerspective(
        image,
        matrix,
        (width, height)
    )

    # Guardar carta rectificada.
    output_image = OUTPUT_DIR / "card_rectified.jpg"

    cv2.imwrite(
        str(output_image),
        warped
    )

    # Guardar imagen de depuración.
    debug_image = DEBUG_DIR / "card_detection.jpg"

    cv2.imwrite(
        str(debug_image),
        detected
    )

    print(f"[SUCCESS] Carta rectificada: {output_image}")

    # Mostrar resultados.
    cv2.imshow(
        "Card Detection",
        detected
    )

    cv2.imshow(
        "Rectified Card",
        warped
    )

    cv2.imshow(
        "Edge Detection",
        edges
    )           

    cv2.waitKey(0)

    cv2.destroyAllWindows()