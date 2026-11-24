import cv2
import json
import numpy as np
from pathlib import Path

import detect_player_indices_v6_2 as v62
import detect_player_indices_v6_4 as v64

ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "config/game_zones.json") as f:
    config = json.load(f)

x1, y1, x2, y2 = config["zones"]["PLAYER_1"]

cam = cv2.VideoCapture(config["camera"]["index"])
cam.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])

if not cam.isOpened():
    raise SystemExit("No se pudo abrir la camara.")

WINDOW = "BLACKJACK - PLAYER V6.2 VS V6.4"

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

congelada = False
frame_guardado = None


def construir_panel(roi, detector, titulo):

    resultado = detector.extraer_indices(roi)
    vista = detector.dibujar_resultados(roi, resultado)

    panel = np.full((750, 580, 3), 25, dtype=np.uint8)

    cv2.putText(
        panel, titulo, (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2
    )

    estado = "CONGELADA" if congelada else "EN VIVO"

    cv2.putText(
        panel,
        f"Indices: {len(resultado['indices'])} | {estado}",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 255, 255), 1
    )

    vista = cv2.resize(vista, (560, 398))
    panel[75:473, 10:570] = vista

    modo = resultado.get("modo_escala", "ESCALERA")
    dimensiones = resultado.get("dimensiones_carta")

    cv2.putText(
        panel, f"Escala: {modo}", (12, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 255, 255), 1
    )

    if dimensiones is not None:
        ancho, alto = dimensiones

        cv2.putText(
            panel,
            f"Carta estimada: {ancho:.1f} x {alto:.1f} px",
            (12, 525),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )

    cv2.putText(
        panel, "RECORTES REALES", (12, 555),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 255), 1
    )

    for i, indice in enumerate(resultado["indices"][:5]):

        recorte = indice["imagen"]
        h, w = recorte.shape[:2]

        escala = min(90 / w, 150 / h)

        nuevo_w = max(1, int(w * escala))
        nuevo_h = max(1, int(h * escala))

        ampliado = cv2.resize(
            recorte,
            (nuevo_w, nuevo_h),
            interpolation=cv2.INTER_NEAREST
        )

        px = 12 + i * 112
        py = 570

        panel[py:py+nuevo_h, px:px+nuevo_w] = ampliado

        cv2.putText(
            panel,
            f"{i+1}: {w}x{h}",
            (px, 740),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (0, 255, 255), 1
        )

    return panel


print("ESPACIO: congelar o reanudar")
print("Q: salir")
print("No se guardaran capturas.")
print("No se modificaran datasets ni modelos.")

try:
    while True:

        if not congelada:
            ok, frame = cam.read()

            if not ok:
                break

            frame_guardado = frame.copy()

        if frame_guardado is None:
            continue

        roi = frame_guardado[y1:y2, x1:x2].copy()

        izquierda = construir_panel(
            roi, v62, "PLAYER V6.2 - ESCALA POR SEPARACION"
        )

        derecha = construir_panel(
            roi, v64, "PLAYER V6.4 - LIMITES Y 35 GRADOS"
        )

        comparacion = np.hstack([izquierda, derecha])

        cv2.imshow(WINDOW, comparacion)

        tecla = cv2.waitKey(30) & 0xFF

        if tecla == ord("q"):
            break

        if tecla == 32:
            congelada = not congelada

finally:
    cam.release()
    cv2.destroyAllWindows()
