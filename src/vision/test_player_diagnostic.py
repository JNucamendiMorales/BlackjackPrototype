import cv2
import json
import numpy as np
from pathlib import Path

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

print("DIAGNOSTICO PLAYER_1")
print("Presiona Q para salir.")

try:
    while True:
        ok, frame = cap.read()

        if not ok:
            break

        zona = frame[y1:y2, x1:x2].copy()

        gris = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
        suavizada = cv2.GaussianBlur(gris, (5, 5), 0)

        _, binaria = cv2.threshold(
            suavizada,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        contornos, _ = cv2.findContours(
            binaria,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidatos = 0

        for contorno in contornos:
            area = cv2.contourArea(contorno)

            if area < 1500:
                continue

            rect = cv2.minAreaRect(contorno)
            ancho, alto = rect[1]

            if min(ancho, alto) <= 0:
                continue

            proporcion = min(ancho, alto) / max(ancho, alto)
            relleno = area / (ancho * alto)

            puntos = cv2.boxPoints(rect).astype(np.int32)

            aceptada = (
                area >= 3000
                and 0.55 <= proporcion <= 0.90
                and relleno >= 0.70
            )

            color = (0, 255, 0) if aceptada else (0, 0, 255)

            cv2.polylines(zona, [puntos], True, color, 2)

            x, y, _, _ = cv2.boundingRect(contorno)

            cv2.putText(
                zona,
                f"A:{area:.0f} R:{proporcion:.2f} F:{relleno:.2f}",
                (max(0, x), max(15, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )

            candidatos += 1

        cv2.putText(
            zona,
            f"Candidatos: {candidatos}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )

        cv2.imshow("PLAYER_1 - DIAGNOSTICO", zona)
        cv2.imshow("PLAYER_1 - BINARIA", binaria)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
