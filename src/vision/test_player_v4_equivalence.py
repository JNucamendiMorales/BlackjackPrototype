"""Compara PLAYER V3.8 y el modulo V4 sobre la misma imagen."""

import json
from pathlib import Path

import cv2
import numpy as np

from detect_player_indices_v4 import extraer_indices, dibujar_resultados

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

print("\nPRUEBA PLAYER V4")
print("Coloca las cinco cartas en escalera.")
print("ESPACIO: congelar imagen")
print("G: guardar resultados")
print("R: reanudar camara")
print("Q: salir\n")

congelada = None
ventana = "PLAYER V4 - VERIFICACION"

try:
    while True:
        if congelada is None:
            ok, frame = cap.read()
            if not ok:
                break

            zona = frame[y1:y2, x1:x2].copy()
        else:
            zona = congelada.copy()

        resultado = extraer_indices(zona)
        vista = dibujar_resultados(zona, resultado)

        ampliada = cv2.resize(
            vista, None, fx=2, fy=2,
            interpolation=cv2.INTER_NEAREST
        )

        cv2.imshow(ventana, ampliada)

        tecla = cv2.waitKey(30) & 0xFF

        if tecla == ord("q"):
            break

        elif tecla == 32:
            if congelada is None:
                congelada = zona.copy()
                print("Imagen congelada.")
                print("Esquinas:", resultado["esquinas"])

        elif tecla == ord("r"):
            congelada = None
            print("Camara reanudada.")

        elif tecla == ord("g"):
            if congelada is None:
                print("Primero presiona ESPACIO.")
                continue

            carpeta = ROOT / "captures" / "player_v4_test"
            carpeta.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(carpeta / "zona_original.png"), zona)
            cv2.imwrite(str(carpeta / "deteccion_v4.png"), vista)

            for indice in resultado["indices"]:
                numero = indice["numero"]
                cv2.imwrite(
                    str(carpeta / f"indice_{numero}.png"),
                    indice["imagen"]
                )

            print("\nRESULTADOS PLAYER V4")
            print("Esquinas:", resultado["esquinas"])
            print("Indices:", len(resultado["indices"]))

            for indice in resultado["indices"]:
                print(
                    f"Indice {indice['numero']}: "
                    f"rectangulo={indice['rectangulo']} "
                    f"resolucion={indice['imagen'].shape[1]}x"
                    f"{indice['imagen'].shape[0]}"
                )

            print(f"\nArchivos guardados en: {carpeta}")

finally:
    cap.release()
    cv2.destroyAllWindows()
