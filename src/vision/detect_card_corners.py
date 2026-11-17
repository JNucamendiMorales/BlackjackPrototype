import cv2
import numpy as np

# ==========================================
# CONFIGURACIÓN
# ==========================================

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MIN_AREA = 5000

# Distancia mínima entre esquinas detectadas
MIN_CORNER_DISTANCE = 25

# ==========================================
# INICIALIZAR CÁMARA
# ==========================================

camera = cv2.VideoCapture(CAMERA_INDEX)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not camera.isOpened():
    print("[ERROR] No se pudo abrir la cámara.")
    raise SystemExit(1)

print("\n===================================")
print(" BLACKJACK - CORNER DETECTION")
print("===================================")
print("[INFO] Cámara conectada.")
print("[INFO] Coloca las cartas diagonalmente.")
print("[INFO] Presiona Q para salir.\n")

# ==========================================
# DETECTAR ESQUINAS
# ==========================================

def detect_corners(binary):

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    corners = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        # Aproximar el contorno exterior
        perimeter = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.01 * perimeter,
            True
        )

        points = approx.reshape(-1, 2)

        # Analizar cambios de dirección
        for i in range(len(points)):

            previous = points[i - 1]
            current = points[i]
            following = points[(i + 1) % len(points)]

            vector1 = current - previous
            vector2 = following - current

            cross = (
                float(vector1[0]) * float(vector2[1])
                - float(vector1[1]) * float(vector2[0])
)

            # Identificar posibles esquinas
            if abs(cross) > 100:

                x, y = current

                # Evitar esquinas demasiado cercanas
                duplicate = False

                for cx, cy in corners:

                    distance = np.sqrt(
                        (x - cx) ** 2 +
                        (y - cy) ** 2
                    )

                    if distance < MIN_CORNER_DISTANCE:
                        duplicate = True
                        break

                if not duplicate:
                    corners.append((int(x), int(y)))

    return corners, contours


# ==========================================
# VIDEO EN TIEMPO REAL
# ==========================================

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("[ERROR] No se pudo capturar el fotograma.")
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

        # Binarización
        _, binary = cv2.threshold(
            blurred,
            120,
            255,
            cv2.THRESH_BINARY
        )

        # Detectar esquinas
        corners, contours = detect_corners(binary)

        # Dibujar contornos
        cv2.drawContours(
            result,
            contours,
            -1,
            (255, 0, 0),
            2
        )

        # Dibujar esquinas
        for i, (x, y) in enumerate(corners):

            cv2.circle(
                result,
                (x, y),
                6,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                result,
                str(i + 1),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Mostrar cantidad
        cv2.putText(
            result,
            f"Esquinas candidatas: {len(corners)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Mostrar ventanas
        cv2.imshow(
            "Blackjack - Corner Detection",
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