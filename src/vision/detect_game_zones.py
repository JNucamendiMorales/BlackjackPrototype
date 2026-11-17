import cv2
import json

from pathlib import Path


# ==========================================
# CONFIGURACIÓN
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FILE = PROJECT_ROOT / "config" / "game_zones.json"


# ==========================================
# CARGAR ZONAS
# ==========================================

if not CONFIG_FILE.exists():

    print("[ERROR] No existe game_zones.json")
    print("[INFO] Ejecuta calibrate_game_zones.py")

    raise SystemExit(1)


with open(CONFIG_FILE, "r", encoding="utf-8") as file:

    configuration = json.load(file)


zones = configuration["zones"]

camera_config = configuration["camera"]


# ==========================================
# INICIALIZAR CÁMARA
# ==========================================

camera = cv2.VideoCapture(
    camera_config["index"]
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    camera_config["width"]
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    camera_config["height"]
)


if not camera.isOpened():

    print("[ERROR] No se pudo abrir la cámara.")

    raise SystemExit(1)


print("\n========================================")
print(" BLACKJACK - GAME ZONE DETECTION")
print("========================================")

print("[SUCCESS] Cámara conectada.")
print("[SUCCESS] Configuración cargada.")

print("\nZONAS ACTIVAS:")

for name, roi in zones.items():

    print(f"{name}: {roi}")

print("\nPresiona Q para salir.\n")


# ==========================================
# PROCESAMIENTO EN TIEMPO REAL
# ==========================================

try:

    while True:

        success, frame = camera.read()

        if not success:

            print("[ERROR] No se pudo capturar imagen.")

            break

        result = frame.copy()

        frame_height, frame_width = frame.shape[:2]

        for name, roi in zones.items():

            x1, y1, x2, y2 = roi

            # Ajustar coordenadas a la imagen real
            x1 = max(0, min(x1, frame_width))
            x2 = max(0, min(x2, frame_width))

            y1 = max(0, min(y1, frame_height))
            y2 = max(0, min(y2, frame_height))

            if x2 <= x1 or y2 <= y1:
                print(f"[WARNING] Zona inválida: {name}")
                continue

            # ==================================
            # EXTRAER REGIÓN
            # ==================================

            region = frame[y1:y2, x1:x2]

            # ==================================
            # DIBUJAR ZONA
            # ==================================

            if name == "PLAYER_1":

                color = (255, 0, 0)

            else:

                color = (0, 255, 0)

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

            # ==================================
            # MOSTRAR REGIÓN INDEPENDIENTE
            # ==================================

            cv2.imshow(
                f"Blackjack - {name}",
                region
            )

        # Mostrar imagen completa
        cv2.imshow(
            "Blackjack - Game Zones",
            result
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

finally:

    camera.release()

    cv2.destroyAllWindows()

    print("[INFO] Cámara liberada.")