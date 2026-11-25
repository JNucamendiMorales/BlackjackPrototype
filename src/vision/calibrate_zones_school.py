import cv2
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

ZONE_NAMES = [
    "PLAYER_1",
    "PLAYER_2",
    "PLAYER_3",
    "DEALER"
]

COLORS = {
    "PLAYER_1": (255, 180, 0),
    "PLAYER_2": (0, 255, 0),
    "PLAYER_3": (0, 180, 255),
    "DEALER": (255, 0, 255)
}

WINDOW = "BLACKJACK - CALIBRACION ESCUELA"

zones = {}


def draw_zones(frame):
    preview = frame.copy()

    for name, roi in zones.items():
        x1, y1, x2, y2 = roi
        color = COLORS[name]

        w = x2 - x1
        h = y2 - y1

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2 - 1, y2 - 1),
            color,
            2
        )

        cv2.putText(
            preview, name,
            (x1 + 5, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 1
        )

        cv2.putText(
            preview, f"{w}x{h}",
            (x1 + 5, y1 + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, color, 1
        )

    return preview


def overlaps(a, b):
    return (
        max(a[0], b[0]) < min(a[2], b[2])
        and
        max(a[1], b[1]) < min(a[3], b[3])
    )


def select_zones(frame):
    previous = zones.copy()
    zones.clear()

    for name in ZONE_NAMES:
        while True:
            preview = draw_zones(frame)

            print(f"\nSelecciona {name}")
            print("Arrastra y presiona ENTER.")

            title = f"Seleccionar {name}"

            x, y, w, h = cv2.selectROI(
                title,
                preview,
                showCrosshair=True,
                fromCenter=False
            )

            cv2.destroyWindow(title)

            if w == 0 or h == 0:
                zones.clear()
                zones.update(previous)
                print("[INFO] Calibracion cancelada.")
                return

            roi = [
                int(x), int(y),
                int(x + w), int(y + h)
            ]

            if any(overlaps(roi, r) for r in zones.values()):
                print("[ERROR] Zonas superpuestas.")
                continue

            zones[name] = roi

            print(f"[OK] {name}: {roi}")
            print(f"Dimensiones: {w} x {h} px")
            print(f"Area: {w * h} px2")

            break


def load_zones(filename):
    path = CONFIG / filename

    if not path.exists():
        print(f"[ERROR] No existe {path}")
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    zones.clear()
    zones.update(data["zones"])

    print(f"[OK] Calibracion cargada: {filename}")


def save_zones():
    if len(zones) != 4:
        print("[ERROR] Faltan zonas.")
        return

    CONFIG.mkdir(parents=True, exist_ok=True)

    version = 1

    while True:
        path = CONFIG / f"game_zones_school_v{version}.json"

        if not path.exists():
            break

        version += 1

    data = {
        "camera": {
            "index": CAMERA_INDEX,
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT
        },
        "zones": zones
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"\n[OK] Nueva calibracion: {path}")

    total = 0

    for name, roi in zones.items():
        x1, y1, x2, y2 = roi
        w = x2 - x1
        h = y2 - y1
        total += w * h

        print(f"{name}: {roi}")
        print(f"Dimensiones: {w} x {h}")
        print(f"Area: {w * h} px2")

    percentage = total / (FRAME_WIDTH * FRAME_HEIGHT) * 100

    print(f"Area total: {total} px2")
    print(f"Aprovechamiento: {percentage:.2f}%")


def main():
    if platform.system() == "Windows":
        camera = cv2.VideoCapture(
            CAMERA_INDEX, cv2.CAP_DSHOW
        )
    else:
        camera = cv2.VideoCapture(CAMERA_INDEX)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not camera.isOpened():
        camera.release()
        raise SystemExit("[ERROR] No se pudo abrir la camara.")

    print("\n========== CALIBRADOR ESCUELA ==========")
    print("C - Calibrar las cuatro zonas")
    print("1 - Cargar calibracion V1")
    print("2 - Cargar calibracion V2")
    print("O - Cargar calibracion oficial")
    print("R - Reiniciar zonas")
    print("S - Guardar como nueva version")
    print("Q - Salir")

    try:
        while True:
            ok, frame = camera.read()

            if not ok:
                print("[ERROR] No se pudo leer la camara.")
                break

            if frame.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
                print("[ERROR] Resolucion inesperada:", frame.shape)
                break

            cv2.imshow(WINDOW, draw_zones(frame))

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                select_zones(frame.copy())

            elif key == ord("1"):
                load_zones("game_zones_school_v1.json")

            elif key == ord("2"):
                load_zones("game_zones_school_v2.json")

            elif key == ord("o"):
                load_zones("game_zones_school.json")

            elif key == ord("r"):
                zones.clear()
                print("[INFO] Zonas reiniciadas.")

            elif key == ord("s"):
                save_zones()

            elif key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
