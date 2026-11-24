"""Reconocimiento de cartas en tiempo real usando OpenCV V2 y CNN V3."""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import detect_dealer_cards_v3_experimental as v2
from train_model_v2 import CardCNNv2


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "blackjack_model_v3.pth"

NOMBRES = {
    "7_T": "7 Treboles",
    "J_C": "J Corazones",
    "K_D": "K Diamantes",
    "Q_C": "Q Corazones",
}


def cargar_modelo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    if checkpoint.get("architecture") != "CardCNNv2":
        raise ValueError("Arquitectura incompatible con CardCNNv2")

    clases = checkpoint["classes"]

    alto, ancho = checkpoint["image_size"]

    transformacion = transforms.Compose([
        transforms.Resize((alto, ancho)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])

    modelo = CardCNNv2(len(clases)).to(device)
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.eval()

    return modelo, clases, transformacion, device


def reconocer_indice(indice, modelo, clases, transformacion, device):
    rgb = cv2.cvtColor(indice, cv2.COLOR_BGR2RGB)
    imagen = Image.fromarray(rgb)

    tensor = transformacion(imagen).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = modelo(tensor)
        probabilidades = torch.softmax(logits, dim=1)[0]
        confianza, posicion = probabilidades.max(dim=0)

    return clases[posicion.item()], confianza.item()


def main():
    modelo, clases, transformacion, device = cargar_modelo()

    print("\nBLACKJACK - RECONOCIMIENTO JUGADOR 1")
    print("Modelo:", MODEL_PATH)
    print("Clases:", clases)
    print("Dispositivo:", device)
    print("Q: salir\n")

    config = v2.original.cam_cfg

    camara = cv2.VideoCapture(config["index"])
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])

    if not camara.isOpened():
        raise SystemExit("No se pudo abrir la camara")

    try:
        while True:
            ok, frame = camara.read()

            if not ok:
                print("No se pudo leer la camara")
                break

            h, w = frame.shape[:2]

            x1, y1, x2, y2 = map(
                int,
                v2.original.zones["PLAYER_1"]
            )

            x1, x2 = sorted((
                max(0, min(w, x1)),
                max(0, min(w, x2))
            ))

            y1, y2 = sorted((
                max(0, min(h, y1)),
                max(0, min(h, y2))
            ))

            dealer = frame[y1:y2, x1:x2].copy()

            _, binaria = v2.original.make_binary(dealer)

            cartas, _ = v2.detectar_reforzado(
                dealer,
                binaria
            )

            pantalla = dealer.copy()

            for numero, (pts, metodo) in enumerate(cartas, 1):
                carta = v2.enderezar(dealer, pts)

                indice = v2.indice(carta, 0.27, 0.49)

                if indice is None or indice.size == 0:
                    continue

                etiqueta, confianza = reconocer_indice(
                    indice,
                    modelo,
                    clases,
                    transformacion,
                    device
                )

                cv2.polylines(
                    pantalla,
                    [pts.astype(np.int32)],
                    True,
                    (0, 255, 0),
                    2
                )

                px, py = pts[0].astype(int)

                nombre = NOMBRES.get(etiqueta, etiqueta)

                texto = f"{nombre} {confianza:.1%}"

                cv2.putText(
                    pantalla,
                    texto,
                    (max(0, px), max(20, py - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

                print(
                    f"\rCarta {numero}: {nombre} | "
                    f"Confianza: {confianza:.1%}",
                    end="",
                    flush=True
                )

            cv2.putText(
                pantalla,
                f"Cartas detectadas: {len(cartas)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )

            cv2.imshow(
                "BLACKJACK - JUGADOR 1 - CNN V3",
                pantalla
            )

            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

    finally:
        camara.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
