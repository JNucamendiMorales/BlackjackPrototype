"""
BlackjackPrototype - Capturador PLAYER V4.

Captura y etiqueta indices de cartas superpuestas en escalera.
No modifica datasets anteriores ni modelos entrenados.

Controles:
    ESPACIO: congelar / reanudar
    G: etiquetar y guardar captura congelada
    Q: salir
"""

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from detect_player_indices_v4 import (
    extraer_indices,
    dibujar_resultados,
)

ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "dataset" / "training_v4_raw"
IMAGES = DATASET / "images"
SESSIONS = DATASET / "sessions"
MANIFEST = DATASET / "manifest.csv"

RANKS = {
    "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "J", "Q", "K", "A"
}

SUITS = {"C", "D", "T", "P"}

WINDOW = "BLACKJACK - CAPTURA PLAYER V4"
SCALE = 2


def validar_etiqueta(texto):
    texto = texto.strip().upper()

    partes = texto.split("_")

    if len(partes) != 2:
        return None

    valor, palo = partes

    if valor not in RANKS or palo not in SUITS:
        return None

    return f"{valor}_{palo}"


def preparar_carpetas():
    IMAGES.mkdir(parents=True, exist_ok=True)
    SESSIONS.mkdir(parents=True, exist_ok=True)


def guardar_captura(zona, resultado, etiquetas, session_id=None):
    """
    Guarda una captura completa y sus indices.
    Todos los indices comparten un mismo capture_id.
    """

    preparar_carpetas()

    marca = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    capture_id = f"{marca}_{uuid.uuid4().hex[:8]}"

    carpeta = SESSIONS / capture_id
    carpeta.mkdir(parents=True, exist_ok=False)

    original_path = carpeta / "player_original.png"
    preview_path = carpeta / "player_deteccion.png"

    cv2.imwrite(str(original_path), zona)

    preview = dibujar_resultados(zona, resultado)
    cv2.imwrite(str(preview_path), preview)

    registros = []

    for indice, etiqueta in zip(resultado["indices"], etiquetas):
        numero = indice["numero"]
        imagen = indice["imagen"]

        nombre = f"{capture_id}_indice_{numero}.png"

        ruta = IMAGES / nombre

        if not cv2.imwrite(str(ruta), imagen):
            raise RuntimeError(f"No se pudo guardar {ruta}")

        registros.append({
            "session_id": session_id or capture_id,
            "capture_id": capture_id,
            "index_number": numero,
            "label": etiqueta,
            "image_path": str(ruta.relative_to(ROOT)),
            "width": imagen.shape[1],
            "height": imagen.shape[0],
            "corner_x": indice["esquina"][0],
            "corner_y": indice["esquina"][1],
        })

    metadata = {
        "capture_id": capture_id,
        "session_id": session_id or capture_id,
        "timestamp": datetime.now().isoformat(),
        "zone": "PLAYER_1",
        "layout": "diagonal_staircase",
        "num_cards": len(etiquetas),
        "labels": etiquetas,
        "indices": [
            {
                "number": indice["numero"],
                "corner": list(indice["esquina"]),
                "rectangle": list(indice["rectangulo"]),
            }
            for indice in resultado["indices"]
        ],
    }

    with open(
        carpeta / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    campos = [
        "session_id",
        "capture_id",
        "index_number",
        "label",
        "image_path",
        "width",
        "height",
        "corner_x",
        "corner_y",
    ]

    existe = MANIFEST.exists() and MANIFEST.stat().st_size > 0

    if existe:
        with open(MANIFEST, newline="", encoding="utf-8") as f:
            columnas_actuales = csv.DictReader(f).fieldnames or []

        if "session_id" not in columnas_actuales:
            manifest_v41 = DATASET / "manifest_v4_1.csv"
        else:
            manifest_v41 = MANIFEST
    else:
        manifest_v41 = MANIFEST

    existe_destino = (
        manifest_v41.exists()
        and manifest_v41.stat().st_size > 0
    )

    with open(manifest_v41, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)

        if not existe_destino:
            writer.writeheader()

        writer.writerows(registros)

    return capture_id, len(registros)


def solicitar_etiquetas(cantidad):
    print("\nETIQUETADO DE CARTAS")
    print("Orden: abajo-izquierda hacia arriba-derecha.")
    print("Palos: C=corazones, D=diamantes, T=treboles, P=picas.")
    print("Ejemplo: J_C, Q_C, 7_T, 7_D, 9_P")
    print("Escribe CANCELAR para no guardar.\n")

    while True:
        texto = input(
            f"Introduce {cantidad} etiquetas separadas por comas: "
        ).strip()

        if texto.upper() == "CANCELAR":
            return None

        etiquetas = [
            validar_etiqueta(x)
            for x in texto.split(",")
        ]

        if len(etiquetas) != cantidad or any(
            etiqueta is None for etiqueta in etiquetas
        ):
            print(
                "[ERROR] Cantidad o formato incorrecto. "
                "Intentalo nuevamente."
            )
            continue

        if len(set(etiquetas)) != len(etiquetas):
            print(
                "[ERROR] Hay cartas duplicadas. "
                "Una baraja fisica solo tiene una de cada tipo."
            )
            continue

        print("\nREVISA LAS ETIQUETAS:")

        for numero, etiqueta in enumerate(etiquetas, start=1):
            print(f"  Indice {numero}: {etiqueta}")

        confirmar = input(
            "\n¿Son correctas? (S/N): "
        ).strip().lower()

        if confirmar == "s":
            return etiquetas

        print("Vamos a introducirlas nuevamente.\n")


def main():
    config_path = ROOT / "config" / "game_zones.json"

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    cam_cfg = config["camera"]

    x1, y1, x2, y2 = map(
        int,
        config["zones"]["PLAYER_1"]
    )

    camara = cv2.VideoCapture(cam_cfg["index"])

    camara.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        cam_cfg["width"]
    )

    camara.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        cam_cfg["height"]
    )

    if not camara.isOpened():
        raise SystemExit("[ERROR] No se pudo abrir la camara.")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    print("\nBLACKJACK - CAPTURA PLAYER V4")
    print("-----------------------------------")
    print("ESPACIO: congelar / continuar")
    print("G: etiquetar y guardar")
    print("Q: salir")
    print("-----------------------------------")
    print("Coloca las cartas superpuestas en escalera.\n")

    congelada = None
    etiquetas_sesion = None
    session_id = None

    try:
        while True:
            if congelada is None:
                ok, frame = camara.read()

                if not ok:
                    print("[ERROR] No se pudo leer la camara.")
                    break

                zona = frame[y1:y2, x1:x2].copy()
            else:
                zona = congelada.copy()

            resultado = extraer_indices(zona)

            vista = dibujar_resultados(zona, resultado)

            estado = (
                "CONGELADA"
                if congelada is not None
                else "EN VIVO"
            )

            cv2.putText(
                vista,
                estado,
                (8, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            ampliada = cv2.resize(
                vista,
                None,
                fx=SCALE,
                fy=SCALE,
                interpolation=cv2.INTER_NEAREST
            )

            cv2.imshow(WINDOW, ampliada)

            tecla = cv2.waitKey(30) & 0xFF

            if tecla == ord("q"):
                break

            elif tecla == 32:
                if congelada is None:
                    congelada = zona.copy()
                    print(
                        f"[OK] Imagen congelada. "
                        f"Indices: {len(resultado['indices'])}"
                    )
                else:
                    congelada = None
                    print("[OK] Camara reanudada.")

            elif tecla == ord("g"):
                if congelada is None:
                    print(
                        "[AVISO] Primero congela "
                        "la imagen con ESPACIO."
                    )
                    continue

                cantidad = len(resultado["indices"])

                if cantidad == 0:
                    print("[AVISO] No se detectaron indices.")
                    continue

                print(
                    f"\n[INFO] Se detectaron {cantidad} indices."
                )

                if (
                    etiquetas_sesion is not None
                    and cantidad == len(etiquetas_sesion)
                ):
                    print("\nETIQUETAS DE LA SESION:")
                    for numero, etiqueta in enumerate(
                        etiquetas_sesion, start=1
                    ):
                        print(f"  Indice {numero}: {etiqueta}")

                    opcion = input(
                        "Usar estas etiquetas? (S/N): "
                    ).strip().lower()

                    if opcion == "s":
                        etiquetas = etiquetas_sesion
                    else:
                        etiquetas = solicitar_etiquetas(cantidad)
                else:
                    etiquetas = solicitar_etiquetas(cantidad)

                if etiquetas is None:
                    print("[AVISO] Captura cancelada.")
                    continue

                if session_id is None or etiquetas != etiquetas_sesion:
                    session_id = uuid.uuid4().hex[:12]

                etiquetas_sesion = etiquetas.copy()

                capture_id, guardados = guardar_captura(
                    zona,
                    resultado,
                    etiquetas,
                    session_id=session_id
                )

                print("\n[OK] CAPTURA GUARDADA")
                print(f"ID: {capture_id}")
                print(f"Indices: {guardados}")
                print(f"Dataset: {DATASET}")
                print(
                    "Puedes reanudar con ESPACIO "
                    "para hacer otra captura.\n"
                )

    finally:
        camara.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
