import cv2


# ==========================================
# CONFIGURACIÓN
# ==========================================

CAMERA_INDEX = 0

# Resolución de la cámara
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Regiones de juego (x1, y1, x2, y2)
PLAYER_ROI = (32, 38, 307, 408)
DEALER_ROI = (333, 38, 608, 408)


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
print("[INFO] JUGADOR: izquierda")
print("[INFO] DEALER: derecha")
print("[INFO] Presiona Q para salir.")


# ==========================================
# FUNCIÓN PARA DIBUJAR REGIONES
# ==========================================

def draw_zone(frame, roi, color, label):

    x1, y1, x2, y2 = roi

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    cv2.putText(
        frame,
        label,
        (x1 + 5, y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


# ==========================================
# VIDEO EN VIVO
# ==========================================

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("[ERROR] No se pudo obtener el fotograma.")
            break

        # Dibujar región del jugador
        draw_zone(
            frame,
            PLAYER_ROI,
            (255, 0, 0),
            "JUGADOR"
        )

        # Dibujar región del dealer
        draw_zone(
            frame,
            DEALER_ROI,
            (0, 255, 0),
            "DEALER"
        )

        # Mostrar resolución
        cv2.putText(
            frame,
            "640 x 480",
            (10, 465),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            "Blackjack - Zone Calibration",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print("[INFO] Cámara liberada.")