"""Reconocimiento en vivo de las ocho cartas del dealer con OpenCV + PyTorch V2.

Ejecutar desde la raíz del proyecto:
    python src/vision/recognize_cards.py

Q: salir | D: alternar diagnóstico | P: pausar/reanudar
"""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

# Importa el detector existente SIN modificarlo.
from detect_dealer_cards import (
    cam_cfg, zones, make_binary, detect_cards,
    straighten_card, extract_card_index,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / 'models' / 'blackjack_model_v2.pth'
EXPECTED_CLASSES = ['4_C', '7_T', '8_T', '9_D', '9_T', 'A_P', 'J_C', 'K_D']
SUITS = {'C': '♥', 'D': '♦', 'T': '♣', 'P': '♠'}


class CardCNNv2(nn.Module):
    """Arquitectura idéntica a train_model_v2.py."""
    def __init__(self, num_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 128), nn.ReLU(),
            nn.Dropout(0.35), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f'No se encontró el modelo: {MODEL_PATH}')
    # weights_only=True evita cargar objetos Python arbitrarios del checkpoint.
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    classes = checkpoint['classes']
    if classes != EXPECTED_CLASSES:
        raise ValueError(f'Clases inesperadas en el modelo: {classes}')
    if checkpoint.get('architecture') != 'CardCNNv2':
        raise ValueError('El checkpoint no corresponde a CardCNNv2.')
    image_size = checkpoint.get('image_size', [128, 128])
    if list(image_size) != [128, 128]:
        raise ValueError(f'Resolución inesperada: {image_size}')
    model = CardCNNv2(len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    return model, classes, transform


def predict(index_bgr, model, classes, transform):
    rgb = cv2.cvtColor(index_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    tensor = transform(image).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    confidence, position = probabilities.max(dim=0)
    return classes[position.item()], float(confidence.item())


def card_label(identifier):
    value, suit = identifier.split('_')
    return f'{value}{SUITS[suit]}'


def dealer_crop(frame):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, zones['DEALER'])
    x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
    y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError('La zona DEALER de config/game_zones.json no es válida.')
    return frame[y1:y2, x1:x2].copy()


def main():
    model, classes, transform = load_model()
    camera = cv2.VideoCapture(cam_cfg['index'])
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg['width'])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg['height'])
    if not camera.isOpened():
        raise SystemExit('No se pudo abrir la cámara. Revisa su índice o cierra otras aplicaciones.')

    print('Reconocimiento PyTorch V2 | Q: salir | D: diagnóstico | P: pausar')
    print('Solo reconoce estas ocho clases:', ', '.join(classes))
    diagnostic = False
    paused = False
    last_frame = None
    try:
        while True:
            if not paused:
                ok, frame = camera.read()
                if not ok:
                    print('No se pudo leer un fotograma de la cámara.')
                    break
                dealer = dealer_crop(frame)
                display = dealer.copy()
                _, binary = make_binary(dealer)
                # detect_cards() imprime una línea por contorno en el programa original;
                # silenciamos solo esos diagnósticos para evitar saturar la terminal.
                with redirect_stdout(StringIO()):
                    cards, _ = detect_cards(dealer, binary)

                results = []
                for corners in cards:
                    corrected = straighten_card(dealer, corners)
                    index = extract_card_index(corrected)
                    if index.size == 0:
                        continue
                    identifier, confidence = predict(index, model, classes, transform)
                    results.append((identifier, confidence))
                    points = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(display, [points], True, (0, 255, 0), 2)
                    x = max(0, int(np.min(points[:, 0, 0])))
                    y = max(20, int(np.min(points[:, 0, 1])) - 8)
                    # OpenCV no dibuja bien símbolos Unicode; usar identificador ASCII.
                    cv2.putText(display, f'{identifier} {confidence:.0%}', (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                lines = [f'{card_label(label)} ({confidence:.1%})' for label, confidence in results]
                if lines:
                    print('\rDetectadas: ' + ' | '.join(lines) + ' ' * 16, end='', flush=True)
                else:
                    print('\rNo se detectaron cartas.                  ', end='', flush=True)

                cv2.imshow('Blackjack - RECONOCIMIENTO V2', display)
                if diagnostic:
                    cv2.imshow('Blackjack - BINARIO', binary)
                last_frame = display
            elif last_frame is not None:
                cv2.imshow('Blackjack - RECONOCIMIENTO V2', last_frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            if key == ord('p'):
                paused = not paused
                print('\nPausado' if paused else '\nReanudado')
            if key == ord('d'):
                diagnostic = not diagnostic
                if not diagnostic:
                    try:
                        cv2.destroyWindow('Blackjack - BINARIO')
                    except cv2.error:
                        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print('\nCámara liberada.')


if __name__ == '__main__':
    main()
