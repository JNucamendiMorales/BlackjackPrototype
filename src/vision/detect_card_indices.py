import cv2
import numpy as np

# ==========================================
# CONFIGURACIÓN
# ==========================================

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MIN_AREA = 1000
MAX_AREA = 150000

# ==========================================
# INICIALIZAR CÁMARA
# ==========================================

camera = cv2.VideoCapture(CAMERA_INDEX)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not camera.isOpened():
    print("[ERROR] No se pudo abrir la cámara.")
    raise SystemExit(1)

print("[INFO] Cámara conectada.")
print("[INFO] Buscando regiones de índices...")
print("[INFO] Presiona Q para salir.")

# ==========================================
# PROCESAMIENTO EN TIEMPO REAL
# ==========================================

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("[ERROR] No se pudo obtener el fotograma.")
            break

        result = frame.copy()

        # Convertir a escala de grises
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Reducir ruido
        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # Separar las cartas blancas del fondo oscuro
        _, binary = cv2.threshold(
            blurred,
            120,
            255,
            cv2.THRESH_BINARY
        )

        # Detectar regiones blancas
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detected_regions = 0

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < MIN_AREA or area > MAX_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Evitar regiones demasiado pequeñas
            if w < 8 or h < 8:
                continue

            detected_regions += 1

            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # Mostrar información
        cv2.putText(
            result,
            f"Regiones detectadas: {detected_regions}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Mostrar ventanas
        cv2.imshow(
            "Blackjack - Index Detection",
            result
        )

        cv2.imshow(
            "Binary Image",
            binary
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print("[INFO] Cámara liberada.")