"""
BlackjackPrototype - Comparacion PLAYER V6 vs V6.1.

Visualiza detecciones y recortes reales.
No guarda capturas.
No modifica datasets ni modelos.
"""

import cv2
import json
import numpy as np
from pathlib import Path

import detect_player_indices_v6_1 as v6
import detect_player_indices_v6_2 as v61

ROOT = Path(__file__).resolve().parents[2]

config = json.loads(
    (ROOT / "config/game_zones.json").read_text()
)

camara = config["camera"]
x1, y1, x2, y2 = config["zones"]["PLAYER_1"]

cap = cv2.VideoCapture(camara["index"])

cap.set(cv2.CAP_PROP_FRAME_WIDTH, camara["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camara["height"])

if not cap.isOpened():
    raise SystemExit("ERROR: No se pudo abrir la camara.")

ventana = "BLACKJACK - PLAYER V6.1 VS V6.2"

cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
cv2.resizeWindow(ventana, 1200, 750)


def crear_panel(roi, detector, titulo, estado):

    resultado = detector.extraer_indices(roi)

    vista = detector.dibujar_resultados(
        roi, resultado
    )

    indices = resultado["indices"]

    ancho = 560
    alto = 650

    panel = np.full(
        (alto, ancho, 3),
        25,
        dtype=np.uint8
    )

    cv2.putText(
        panel,
        titulo,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        panel,
        f"Indices: {len(indices)} | {estado}",
        (15, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1
    )

    vista = cv2.resize(
        vista,
        (530, 377),
        interpolation=cv2.INTER_NEAREST
    )

    panel[70:447, 15:545] = vista

    cv2.putText(
        panel,
        "RECORTES REALES",
        (15, 477),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    for j, indice in enumerate(indices[:5]):

        recorte = indice["imagen"]

        h, w = recorte.shape[:2]

        escala = min(85 / w, 125 / h)

        ampliado = cv2.resize(
            recorte,
            (
                max(1, int(w * escala)),
                max(1, int(h * escala))
            ),
            interpolation=cv2.INTER_NEAREST
        )

        ah, aw = ampliado.shape[:2]

        px = 15 + j * 108
        py = 500

        panel[py:py + ah, px:px + aw] = ampliado

        cv2.putText(
            panel,
            f"{indice['numero']}: {w}x{h}",
            (px, 644),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 255),
            1
        )

    return panel, len(indices)


congelada = False
frame_actual = None

print("========================================")
print("COMPARACION PLAYER V6.1 VS V6.2")
print("Izquierda: V6.1 - Recorte original")
print("Derecha: V6.2 - Recorte ampliado")
print("Parte inferior: Recortes reales")
print("ESPACIO: Congelar / Reanudar")
print("Q: Salir")
print("========================================")

try:

    while True:

        if not congelada:

            ok, frame = cap.read()

            if not ok:
                break

            frame_actual = frame.copy()

        if frame_actual is None:
            continue

        roi = frame_actual[y1:y2, x1:x2].copy()

        estado = (
            "CONGELADA" if congelada else "EN VIVO"
        )

        panel_v6, cantidad_v6 = crear_panel(
            roi,
            v6,
            "PLAYER V6.1 - RECORTE ORIGINAL",
            estado
        )

        panel_v61, cantidad_v61 = crear_panel(
            roi,
            v61,
            "PLAYER V6.2 - RECORTE AMPLIADO",
            estado
        )

        comparacion = np.hstack([
            panel_v6,
            panel_v61
        ])

        cv2.imshow(ventana, comparacion)

        tecla = cv2.waitKey(20) & 0xFF

        if tecla == ord("q"):
            break

        if tecla == 32:

            congelada = not congelada

            if congelada:
                print(
                    f"[PRUEBA] V6.1: {cantidad_v6} indices"
                    f" | V6.2: {cantidad_v61} indices"
                )

finally:

    cap.release()
    cv2.destroyAllWindows()

