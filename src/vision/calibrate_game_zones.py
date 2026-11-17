import cv2
import json
from pathlib import Path

# ==========================================
# CONFIGURACIÓN
# ==========================================

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = CONFIG_DIR / "game_zones.json"

WINDOW_NAME = "Blackjack - Zone Calibration"

ZONE_NAMES = ["PLAYER_1", "DEALER"]

ZONE_COLORS = {
    "PLAYER_1": (255, 0, 0),
    "DEALER": (0, 255, 0)
}

# ==========================================
# VARIABLES
# ==========================================

zones = {}

current_zone = None
drawing = False

start_point = None
end_point = None


# ==========================================
# FUNCIÓN DEL MOUSE
# ==========================================

def mouse_callback(event, x, y, flags, param):

    global drawing
    global start_point
    global end_point
    global current_zone

    if current_zone is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:

        if drawing:

            drawing = False
            end_point = (x, y)

            x1 = min(start_point[0], end_point[0])
            y1 = min(start_point[1], end_point[1])

            x2 = max(start_point[0], end_point[0])
            y2 = max(start_point[1], end_point[1])

            if x2 - x1 < 20 or y2 - y1 < 20:

                print("[WARNING] Zona demasiado pequeña.")
                return

            zones[current_zone] = [x1, y1, x2, y2]

            print(
                f"[SUCCESS] {current_zone}: "
                f"{zones[current_zone]}"
            )

            current_zone = None


# ==========================================
# GUARDAR CONFIGURACIÓN
# ==========================================

def save_zones():

    if not all(name in zones for name in ZONE_NAMES):

        print("[ERROR] Debes configurar ambas zonas.")
        return

    configuration = {
        "camera": {
            "index": CAMERA_INDEX,
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT
        },
        "zones": zones
    }

    with open(CONFIG_FILE, "w", encoding="utf-8") as file:

        json.dump(configuration, file, indent=4)

    print("\n===================================")
    print(" CONFIGURACIÓN GUARDADA")
    print("===================================")
    print(f"Archivo: {CONFIG_FILE}")


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
print(" BLACKJACK - ZONE CALIBRATION")
print("===================================")

print("[INFO] Cámara conectada correctamente.")

print("\nCONTROLES:")
print("1 - Seleccionar zona del jugador")
print("2 - Seleccionar zona del dealer")
print("S - Guardar configuración")
print("R - Reiniciar zonas")
print("Q - Salir\n")

cv2.namedWindow(WINDOW_NAME)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


# ==========================================
# VIDEO EN TIEMPO REAL
# ==========================================

try:

    while True:

        success, frame = camera.read()

        if not success:

            print("[ERROR] No se pudo obtener el fotograma.")
            break

        result = frame.copy()

        # Dibujar zonas seleccionadas
        for name, roi in zones.items():

            x1, y1, x2, y2 = roi

            color = ZONE_COLORS[name]

            cv2.rectangle(
                result,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                result,
                name,
                (x1 + 5, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # Dibujar rectángulo mientras seleccionamos
        if drawing and start_point and end_point:

            cv2.rectangle(
                result,
                start_point,
                end_point,
                (0, 255, 255),
                2
            )

        # Mostrar instrucciones
        if current_zone is not None:

            cv2.putText(
                result,
                f"Seleccionando: {current_zone}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        else:

            cv2.putText(
                result,
                "1: Jugador | 2: Dealer | S: Guardar",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        cv2.imshow(WINDOW_NAME, result)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):

            current_zone = "PLAYER_1"

            print("[INFO] Selecciona la zona del jugador.")

        elif key == ord("2"):

            current_zone = "DEALER"

            print("[INFO] Selecciona la zona del dealer.")

        elif key == ord("s"):

            save_zones()

        elif key == ord("r"):

            zones.clear()

            current_zone = None
            drawing = False

            print("[INFO] Zonas reiniciadas.")

        elif key == ord("q"):

            break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print("[INFO] Cámara liberada correctamente.")