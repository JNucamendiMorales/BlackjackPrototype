"""Calibración de cartas superpuestas para PLAYER_1."""

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/game_zones.json"
OUTPUT = ROOT / "config/player_overlap_calibration.json"

with CONFIG.open(encoding="utf-8") as f:
    config = json.load(f)

cam = config["camera"]
x1, y1, x2, y2 = config["zones"]["PLAYER_1"]

points = []
frozen = None


def mouse_callback(event, x, y, flags, userdata):
    global points

    if event == cv2.EVENT_LBUTTONDOWN and frozen is not None:
        if len(points) < 4:
            points.append((x, y))
            print(f"Punto {len(points)}: ({x}, {y})")


def draw_points(image):
    result = image.copy()

    labels = [
        "Carta 1 - esquina superior izquierda",
        "Carta 1 - esquina inferior derecha",
        "Carta 2 - esquina superior izquierda",
        "Carta 2 - esquina inferior derecha"
    ]

    for i, (x, y) in enumerate(points):
        cv2.circle(result, (x, y), 5, (0, 0, 255), -1)

        cv2.putText(
            result,
            f"P{i+1}",
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

    if frozen is not None and len(points) < 4:
        cv2.putText(
            result,
            f"Marca P{len(points)+1}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

    return result


cap = cv2.VideoCapture(cam["index"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])

if not cap.isOpened():
    raise SystemExit("No se pudo abrir la camara.")

cv2.namedWindow("CALIBRACION PLAYER V3")
cv2.setMouseCallback("CALIBRACION PLAYER V3", mouse_callback)

print("\nCALIBRACION DE CARTAS SUPERPUESTAS")
print("1. Coloca dos cartas superpuestas.")
print("2. Presiona ESPACIO para congelar.")
print("3. Marca cuatro puntos:")
print("   P1: esquina superior izquierda de carta 1")
print("   P2: esquina inferior derecha de carta 1")
print("   P3: esquina superior izquierda de carta 2")
print("   P4: esquina inferior derecha de carta 2")
print("4. Presiona G para guardar.")
print("R: reiniciar | Q: salir\n")

try:
    while True:
        if frozen is None:
            ok, frame = cap.read()

            if not ok:
                break

            zona = frame[y1:y2, x1:x2].copy()
        else:
            zona = frozen.copy()

        display = draw_points(zona)

        cv2.imshow("CALIBRACION PLAYER V3", display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

        if key == 32:
            frozen = zona.copy()
            points = []
            print("Imagen congelada. Marca los cuatro puntos.")

        if key == ord("r"):
            frozen = None
            points = []
            print("Calibracion reiniciada.")

        if key == ord("g"):
            if len(points) != 4:
                print("Debes marcar los cuatro puntos.")
                continue

            p1, p2, p3, p4 = points

            width1 = abs(p2[0] - p1[0])
            height1 = abs(p2[1] - p1[1])

            dx = p3[0] - p1[0]
            dy = p3[1] - p1[1]

            data = {
                "timestamp": datetime.now().isoformat(),
                "zone": "PLAYER_1",
                "reference_card": {
                    "width_px": width1,
                    "height_px": height1
                },
                "overlap": {
                    "dx_px": dx,
                    "dy_px": dy,
                    "dx_relative": dx / width1 if width1 else 0,
                    "dy_relative": dy / height1 if height1 else 0
                },
                "points": points
            }

            with OUTPUT.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            print("\nCALIBRACION GUARDADA")
            print(f"Ancho de carta: {width1} px")
            print(f"Alto de carta: {height1} px")
            print(f"Desplazamiento horizontal: {dx} px")
            print(f"Desplazamiento vertical: {dy} px")
            print(f"Archivo: {OUTPUT}")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
