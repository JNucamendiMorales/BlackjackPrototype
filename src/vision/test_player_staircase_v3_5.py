"""
PLAYER V3.1
Deteccion experimental de esquinas superiores izquierdas
en cartas superpuestas en escalera diagonal.

No modifica los detectores anteriores.
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

WINDOW = "PLAYER V3.5 - INDICES AJUSTADOS"
SCALE = 2


def detectar_esquinas(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)

    _, mascara = cv2.threshold(
        gris, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos:
        return [], None

    contorno = max(contornos, key=cv2.contourArea)

    if cv2.contourArea(contorno) < 3000:
        return [], None

    perimetro = cv2.arcLength(contorno, True)

    aproximacion = cv2.approxPolyDP(
        contorno,
        0.012 * perimetro,
        True
    )

    puntos = aproximacion.reshape(-1, 2)

    candidatos = []

    # Una esquina superior izquierda tiene un tramo horizontal
    # hacia la derecha y un tramo vertical hacia abajo.
    # Analizamos vecinos del poligono cerrado.
    n = len(puntos)

    for i in range(n):
        actual = puntos[i]
        anterior = puntos[(i - 1) % n]
        siguiente = puntos[(i + 1) % n]

        vecinos = [anterior, siguiente]

        derecha = any(
            p[0] > actual[0] + 8
            and abs(p[1] - actual[1]) < 12
            for p in vecinos
        )

        abajo = any(
            p[1] > actual[1] + 8
            and abs(p[0] - actual[0]) < 12
            for p in vecinos
        )

        if derecha and abajo:
            candidatos.append(tuple(map(int, actual)))

    candidatos.sort(key=lambda p: p[0])

    # Seleccionar la cadena ascendente hacia la derecha.
    # Cada nueva esquina debe quedar arriba de la anterior.
    filtrados = []

    for punto in candidatos:
        if not filtrados:
            filtrados.append(punto)
            continue

        ultimo = filtrados[-1]

        dx = punto[0] - ultimo[0]
        dy = punto[1] - ultimo[1]

        if dx > 5 and dy < -5:
            filtrados.append(punto)

    return filtrados, contorno


cap = cv2.VideoCapture(cam_cfg["index"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])

if not cap.isOpened():
    raise SystemExit("No se pudo abrir la camara.")

cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

print("\nPLAYER V3.1")
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

        esquinas, contorno = detectar_esquinas(zona)

        display = zona.copy()

        if contorno is not None:
            cv2.drawContours(
                display,
                [contorno],
                -1,
                (0, 255, 0),
                2
            )

        for i, (px, py) in enumerate(esquinas):

            # Region experimental del indice superior izquierdo.
            # Valores iniciales; se ajustaran con las capturas.
            # Estimamos el tamaño del escalon usando
            # la separacion horizontal entre esquinas.
            separaciones = [
                esquinas[j + 1][0] - esquinas[j][0]
                for j in range(len(esquinas) - 1)
                if esquinas[j + 1][0] > esquinas[j][0]
            ]

            if separaciones:
                ancho_escalon = float(np.median(separaciones))
            else:
                ancho_escalon = 35.0

            # Ajustar el recorte cerca de la esquina real.
            margen_x = 1
            margen_y = 1

            ancho_indice = max(12, int(ancho_escalon * 0.48))
            alto_indice = max(15, int(ancho_escalon * 0.65))

            rx1 = max(0, px + margen_x)
            ry1 = max(0, py + margen_y)

            rx2 = min(display.shape[1], rx1 + ancho_indice)
            ry2 = min(display.shape[0], ry1 + alto_indice)

            cv2.rectangle(
                display,
                (rx1, ry1),
                (rx2, ry2),
                (255, 0, 0),
                2
            )

            cv2.circle(
                display,
                (px, py),
                6,
                (255, 0, 255),
                -1
            )

            cv2.putText(
                display,
                str(i + 1),
                (px + 7, py - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        cv2.putText(
            display,
            f"Esquinas candidatas: {len(esquinas)}",
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
