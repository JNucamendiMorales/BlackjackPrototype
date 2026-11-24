"""
BlackjackPrototype - Comparacion PLAYER V5 vs V6.

No guarda capturas.
No modifica datasets.
No modifica modelos.
"""

import cv2
import json
import numpy as np
from pathlib import Path

import detect_player_indices_v5 as v5
import detect_player_indices_v6 as v6

ROOT = Path(__file__).resolve().parents[2]

config = json.loads(
    (ROOT / "config/game_zones.json").read_text()
)

camara = config["camera"]
x1, y1, x2, y2 = config["zones"]["PLAYER_1"]

cap = cv2.VideoCapture(camara["index"])

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    camara["width"]
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    camara["height"]
)

if not cap.isOpened():
    raise SystemExit("ERROR: No se pudo abrir la camara.")

ventana = "BLACKJACK - PLAYER V5 VS V6"

cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
cv2.resizeWindow(ventana, 1200, 500)

congelada = False
frame_actual = None

print("========================================")
print("COMPARACION PLAYER V5 VS V6")
print("Izquierda: V5")
print("Derecha: V6")
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

        resultado_v5 = v5.extraer_indices(roi)
        resultado_v6 = v6.extraer_indices(roi)

        vista_v5 = v5.dibujar_resultados(
            roi, resultado_v5
        )

        vista_v6 = v6.dibujar_resultados(
            roi, resultado_v6
        )

        cantidad_v5 = len(resultado_v5["indices"])
        cantidad_v6 = len(resultado_v6["indices"])

        estado = (
            "CONGELADA" if congelada else "EN VIVO"
        )

        paneles = []

        for vista, titulo, cantidad in [
            (vista_v5, "PLAYER V5", cantidad_v5),
            (vista_v6, "PLAYER V6 - 25 grados", cantidad_v6)
        ]:

            panel = cv2.copyMakeBorder(
                vista,
                65, 10, 10, 10,
                cv2.BORDER_CONSTANT,
                value=(25, 25, 25)
            )

            cv2.putText(
                panel,
                titulo,
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

            cv2.putText(
                panel,
                f"Indices: {cantidad} | {estado}",
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )

            paneles.append(panel)

        comparacion = np.hstack(paneles)

        cv2.imshow(ventana, comparacion)

        tecla = cv2.waitKey(20) & 0xFF

        if tecla == ord("q"):
            break

        if tecla == 32:

            congelada = not congelada

            if congelada:
                print(
                    f"[PRUEBA] V5: {cantidad_v5} indices"
                    f" | V6: {cantidad_v6} indices"
                )

finally:

    cap.release()
    cv2.destroyAllWindows()

