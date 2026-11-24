import cv2
import json
from pathlib import Path
from detect_player_cards_v2 import detectar_cartas

ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "config/game_zones.json", encoding="utf-8") as f:
    config = json.load(f)

cam = config["camera"]
x1, y1, x2, y2 = config["zones"]["PLAYER_1"]

cap = cv2.VideoCapture(cam["index"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])

if not cap.isOpened():
    raise SystemExit("No se pudo abrir la cámara.")

print("DETECTOR PLAYER_1 V1")
print("Presiona Q para salir.")

try:
    while True:
        ok, frame = cap.read()

        if not ok:
            break

        zona = frame[y1:y2, x1:x2].copy()
        cartas = detectar_cartas(zona)

        for i, (puntos, metodo) in enumerate(cartas, 1):
            puntos = puntos.astype("int32")

            cv2.polylines(
                zona, [puntos], True, (0, 255, 0), 2
            )

            px, py = puntos[0]

            cv2.putText(
                zona,
                f"Carta {i}",
                (max(0, int(px)), max(20, int(py) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        cv2.putText(
            zona,
            f"Cartas detectadas: {len(cartas)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.imshow("PLAYER_1 - DETECTOR V2", zona)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
