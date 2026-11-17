import cv2
import numpy as np
import time

from pathlib import Path
from datetime import datetime

# ==========================================
# IMPORTAR FUNCIONES DEL DETECTOR
# ==========================================

from detect_dealer_cards import (
    cam_cfg,
    zones,
    make_binary,
    detect_cards,
    straighten_card,
    extract_card_index
)

# ==========================================
# CONFIGURACION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "dataset" / "training"

DATASET.mkdir(parents=True, exist_ok=True)

VALID_RANKS = [
    "2", "3", "4", "5", "6", "7",
    "8", "9", "10", "J", "Q", "K", "A"
]

VALID_SUITS = ["C", "D", "T", "P"]

# ==========================================
# CONFIGURACION DE CAPTURA AUTOMATICA
# ==========================================

CAPTURE_INTERVAL = 4.0

MAX_AUTO_CAPTURES = 30

auto_capture = False

auto_count = 0

last_capture_time = 0

# ==========================================
# SELECCIONAR CARTA
# ==========================================

print("\n====================================")
print(" BLACKJACK - CAPTURA DE INDICES")
print("====================================")

print("\nPALOS:")
print("C = Corazones")
print("D = Diamantes")
print("T = Treboles")
print("P = Picas")

rank = input(
    "\nNumero o letra de la carta: "
).upper().strip()

suit = input(
    "Palo de la carta: "
).upper().strip()

if rank not in VALID_RANKS:
    raise SystemExit(
        "[ERROR] Numero de carta no valido."
    )

if suit not in VALID_SUITS:
    raise SystemExit(
        "[ERROR] Palo de carta no valido."
    )

card_name = f"{rank}_{suit}"

SAVE_DIR = DATASET / card_name

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print(f"\nCapturando indices de: {card_name}")

# ==========================================
# FUNCION PARA GUARDAR IMAGEN
# ==========================================

def save_index(index):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = (
        SAVE_DIR /
        f"{card_name}_{timestamp}.png"
    )

    saved = cv2.imwrite(
        str(filename),
        index
    )

    if saved:

        print(
            f"[OK] Imagen guardada: {filename.name}"
        )

    else:

        print(
            "[ERROR] No se pudo guardar la imagen."
        )

    return saved


# ==========================================
# INICIAR CAMARA
# ==========================================

camera = cv2.VideoCapture(
    cam_cfg["index"]
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    cam_cfg["width"]
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    cam_cfg["height"]
)

if not camera.isOpened():

    raise SystemExit(
        "[ERROR] No se pudo abrir la camara."
    )

print("\nCONTROLES:")
print("S = Guardar una imagen")
print("A = Iniciar / detener captura automatica")
print("Q = Salir")

print("\nColoca UNA carta en la zona DEALER.")

# ==========================================
# BUCLE PRINCIPAL
# ==========================================

try:

    while True:

        ok, frame = camera.read()

        if not ok:

            print(
                "[ERROR] No se pudo leer la camara."
            )

            break

        # ==========================================
        # OBTENER ZONA DEALER
        # ==========================================

        h, w = frame.shape[:2]

        x1, y1, x2, y2 = map(
            int,
            zones["DEALER"]
        )

        x1, x2 = sorted((
            max(0, min(w, x1)),
            max(0, min(w, x2))
        ))

        y1, y2 = sorted((
            max(0, min(h, y1)),
            max(0, min(h, y2))
        ))

        if x2 <= x1 or y2 <= y1:

            print(
                "[ERROR] Zona DEALER invalida."
            )

            break

        dealer = frame[
            y1:y2,
            x1:x2
        ].copy()

        display = dealer.copy()

        # ==========================================
        # DETECTAR CARTAS
        # ==========================================

        threshold_value, binary = make_binary(
            dealer
        )

        cards, contours = detect_cards(
            dealer,
            binary
        )
        
        # ==========================================
        # DIAGNOSTICO DE DETECCION
        # ==========================================

        debug = cv2.cvtColor(
            binary,
            cv2.COLOR_GRAY2BGR
        )

        cv2.drawContours(
            debug,
            contours,
            -1,
            (0, 255, 0),
            1
        )

        cv2.imshow(
            "Blackjack - DIAGNOSTICO BINARIO",
            debug
        )
        
        

        # ==========================================
        # EXTRAER INDICE
        # ==========================================

        index = None

        if len(cards) == 1:

            corners = cards[0]

            cv2.polylines(
                display,
                [corners.astype(np.int32)],
                True,
                (0, 255, 0),
                2
            )

            corrected_card = straighten_card(
                dealer,
                corners
            )

            index = extract_card_index(
                corrected_card
            )

            if index.size == 0:

                index = None

        # ==========================================
        # INFORMACION EN PANTALLA
        # ==========================================

        cv2.putText(
            display,
            f"Carta: {card_name}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Detectadas: {len(cards)}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # ==========================================
        # ESTADO DE CAPTURA
        # ==========================================

        if auto_capture:

            status = (
                f"AUTO: {auto_count}/"
                f"{MAX_AUTO_CAPTURES}"
            )

            color = (0, 255, 0)

        else:

            status = "AUTO: DETENIDO"

            color = (0, 0, 255)

        cv2.putText(
            display,
            status,
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        # ==========================================
        # PREPARAR VISTA PREVIA
        # ==========================================

        if index is not None:

            index_preview = cv2.resize(
                index,
                (150, 200),
                interpolation=cv2.INTER_CUBIC
            )

        else:

            index_preview = np.zeros(
                (200, 150, 3),
                dtype=np.uint8
            )

        # ==========================================
        # MOSTRAR VENTANAS
        # ==========================================

        cv2.imshow(
            "Blackjack - CAPTURA DEALER",
            display
        )

        cv2.imshow(
            "Blackjack - INDICE A GUARDAR",
            index_preview
        )

        # ==========================================
        # CAPTURA AUTOMATICA
        # ==========================================

        current_time = time.monotonic()

        if auto_capture:

            elapsed = (
                current_time - last_capture_time
            )

            if elapsed >= CAPTURE_INTERVAL:

                if index is not None:

                    saved = save_index(index)

                    if saved:

                        auto_count += 1

                        last_capture_time = current_time

                else:

                    # Esperar a que vuelva a detectarse
                    # una sola carta correctamente.
                    last_capture_time = current_time

            if auto_count >= MAX_AUTO_CAPTURES:

                auto_capture = False

                print(
                    "\n[OK] Captura automatica completada."
                )

        # ==========================================
        # CONTROLES
        # ==========================================

        key = cv2.waitKey(30) & 0xFF

        if key == ord("s"):

            if index is not None:

                save_index(index)

            else:

                print(
                    "[AVISO] Debe detectarse "
                    "exactamente una carta."
                )

        elif key == ord("a"):

            auto_capture = not auto_capture

            if auto_capture:

                auto_count = 0

                last_capture_time = (
                    time.monotonic()
                )

                print(
                    "\n[OK] Captura automatica iniciada."
                )

            else:

                print(
                    "\n[OK] Captura automatica detenida."
                )

        elif key == ord("q"):

            break

finally:

    camera.release()

    cv2.destroyAllWindows()