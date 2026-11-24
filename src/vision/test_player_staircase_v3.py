"""
PLAYER V3 - Experimento de deteccion de esquinas.

Objetivo:
Encontrar posibles esquinas superiores izquierdas
de cartas superpuestas en escalera diagonal.

No modifica los detectores existentes.
"""

import cv2
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "config/game_zones.json", encoding="utf-8") as f:
    config = json.load(f)

cam_cfg = config["camera"]
x1, y1, x2, y2 = config["zones"]["PLAYER_1"]

cap = cv2.VideoCapture(cam_cfg["index"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])

if not cap.isOpened():
    raise SystemExit("No se pudo abrir la camara.")

WINDOW = "PLAYER V3 - ESCALERA"
SCALE = 2

cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

print("\nPLAYER V3 - PRUEBA DE ESQUINAS")
print("Coloca las cartas en escalera diagonal.")
print("ESPACIO: congelar / continuar")
print("Q: salir\n")

frozen = None

try:
    while True:
        if frozen is None:
            ok, frame = cap.read()

            if not ok:
                break

            zona = frame[y1:y2, x1:x2].copy()
        else:
            zona = frozen.copy()

        gris = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
        gris = cv2.GaussianBlur(gris, (5, 5), 0)

        _, mascara = cv2.threshold(
            gris,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        contornos, _ = cv2.findContours(
            mascara,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        display = zona.copy()
        candidatos = []

        for contorno in contornos:
            if cv2.contourArea(contorno) < 3000:
                continue

            cv2.drawContours(
                display,
                [contorno],
                -1,
                (0, 255, 0),
                2
            )

            perimetro = cv2.arcLength(contorno, True)

            aproximacion = cv2.approxPolyDP(
                contorno,
                0.012 * perimetro,
                True
            )

            for punto in aproximacion.reshape(-1, 2):
                px, py = map(int, punto)

                candidatos.append((px, py))

                cv2.circle(
                    display,
                    (px, py),
                    4,
                    (0, 0, 255),
                    -1
                )

        cv2.putText(
            display,
            f"Vertices del contorno: {len(candidatos)}",
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2
        )

        ampliada = cv2.resize(
            display,
            None,
            fx=SCALE,
            fy=SCALE,
            interpolation=cv2.INTER_NEAREST
        )

        cv2.imshow(WINDOW, ampliada)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

        if key == 32:
            if frozen is None:
                frozen = zona.copy()
                print("Imagen congelada.")
            else:
                frozen = None
                print("Camara reanudada.")

finally:
    cap.release()
    cv2.destroyAllWindows()
