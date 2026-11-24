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
    raise SystemExit("No se pudo abrir la camara.")

print("PRUEBA DE VERTICES PLAYER_1")
print("VERDE: vertices reales")
print("AZUL: rectangulo rotado")
print("Q: salir")

try:
    while True:
        ok, frame = cap.read()

        if not ok:
            break

        zona = frame[y1:y2, x1:x2].copy()

        gris = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
        gris = cv2.GaussianBlur(gris, (5, 5), 0)

        _, binaria = cv2.threshold(
            gris, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        contornos, _ = cv2.findContours(
            binaria,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contorno in contornos:
            if cv2.contourArea(contorno) < 3000:
                continue

            rect = cv2.minAreaRect(contorno)
            caja = cv2.boxPoints(rect).astype(np.int32)

            cv2.polylines(
                zona, [caja], True, (255, 0, 0), 2
            )

            casco = cv2.convexHull(contorno)
            perimetro = cv2.arcLength(casco, True)

            aproximacion = cv2.approxPolyDP(
                casco, 0.02 * perimetro, True
            )

            if len(aproximacion) == 4:
                cv2.polylines(
                    zona,
                    [aproximacion],
                    True,
                    (0, 255, 0),
                    2
                )

                for i, punto in enumerate(aproximacion):
                    px, py = punto[0]

                    cv2.circle(
                        zona, (px, py), 5,
                        (0, 0, 255), -1
                    )

                    cv2.putText(
                        zona,
                        str(i + 1),
                        (px + 5, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2
                    )

        cv2.imshow("PLAYER_1 - VERTICES", zona)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
