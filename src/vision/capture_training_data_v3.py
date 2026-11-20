"""Captura de índices para CNN V3 usando detector de cuatro vértices.

Ejecutar desde la raíz del proyecto:
    python src/vision/capture_training_data_v3.py

No modifica el detector ni el dataset anterior.
"""
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import detect_dealer_cards_v2_mejorado as v2

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / 'dataset' / 'training_v3'
EVIDENCE = ROOT / 'dataset' / 'debug_samples' / 'captura_v3'
VALID_RANKS = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
VALID_SUITS = ['C','D','T','P']
CAPTURE_INTERVAL = 4.0
MAX_AUTO_CAPTURES = 30


def seleccionar_carta():
    print('\nBLACKJACK - CAPTURA DE INDICES V3 (4 VERTICES)')
    print('Palos: C=Corazones, D=Diamantes, T=Treboles, P=Picas')
    rank = input('Numero o letra de la carta: ').upper().strip()
    suit = input('Palo de la carta: ').upper().strip()
    if rank not in VALID_RANKS or suit not in VALID_SUITS:
        raise SystemExit('[ERROR] Valor o palo no valido.')
    return f'{rank}_{suit}'


def zona_dealer(frame):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, v2.original.zones['DEALER'])
    x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
    y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Zona DEALER invalida')
    return frame[y1:y2, x1:x2].copy()


def extraer_una_carta(dealer):
    # Se conserva la misma binarizacion y deteccion reforzada del detector V2.
    _, binaria = v2.original.make_binary(dealer)
    cartas, alternativa = v2.detectar_reforzado(dealer, binaria)
    if len(cartas) != 1:
        return None, None, None, len(cartas), alternativa
    pts, metodo = cartas[0]
    corregida = v2.enderezar(dealer, pts)
    if corregida is None or corregida.size == 0:
        return None, None, None, len(cartas), alternativa
    idx = v2.indice(corregida, 0.27, 0.49)
    if idx is None or idx.size == 0:
        return None, None, None, len(cartas), alternativa
    return idx, corregida, pts, len(cartas), metodo


def guardar(imagen, carpeta, prefijo):
    carpeta.mkdir(parents=True, exist_ok=True)
    marca = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    ruta = carpeta / f'{prefijo}_{marca}.png'
    if not cv2.imwrite(str(ruta), imagen):
        print(f'[ERROR] No se pudo guardar {ruta}')
        return None
    print(f'[OK] {ruta.relative_to(ROOT)}')
    return ruta


def main():
    nombre = seleccionar_carta()
    carpeta = DATASET / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    cam_cfg = v2.original.cam_cfg
    camara = cv2.VideoCapture(cam_cfg['index'])
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg['width'])
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg['height'])
    if not camara.isOpened():
        raise SystemExit('[ERROR] No se pudo abrir la camara')
    automatico = False
    contador = 0
    ultima = time.monotonic()
    print(f'\nClase: {nombre} | Guardando en: {carpeta}')
    print('Coloca UNA carta separada en la zona DEALER.')
    print('S: guardar indice | A: auto 30 imagenes / 4s | E: evidencia | Q: salir')
    print('IMPORTANTE: confirma que la vista previa muestra la carta etiquetada.')
    try:
        while True:
            ok, frame = camara.read()
            if not ok:
                print('[ERROR] No se pudo leer la camara')
                break
            dealer = zona_dealer(frame)
            vista = dealer.copy()
            idx, corregida, pts, total, metodo = extraer_una_carta(dealer)
            if pts is not None:
                cv2.polylines(vista, [np.asarray(pts, dtype=np.int32)], True, (0, 255, 0), 2)
            cv2.putText(vista, f'{nombre} | Detectadas: {total}', (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            estado = f'AUTO {contador}/{MAX_AUTO_CAPTURES}' if automatico else 'AUTO detenido'
            cv2.putText(vista, estado, (10, 53), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0) if automatico else (0, 0, 255), 2)
            previa = (cv2.resize(idx, (162, 220), interpolation=cv2.INTER_CUBIC)
                      if idx is not None else np.zeros((220, 162, 3), dtype=np.uint8))
            cv2.imshow('V3 - DEALER', vista)
            cv2.imshow('V3 - INDICE 27% (SE GUARDA)', previa)
            if corregida is not None:
                cv2.imshow('V3 - CARTA ENDEREZADA', corregida)
            ahora = time.monotonic()
            if automatico and ahora - ultima >= CAPTURE_INTERVAL:
                if idx is not None and guardar(idx, carpeta, nombre):
                    contador += 1
                ultima = ahora
                if contador >= MAX_AUTO_CAPTURES:
                    automatico = False
                    print('[OK] Captura automatica completada.')
            tecla = cv2.waitKey(30) & 0xFF
            if tecla == ord('s'):
                if idx is None:
                    print('[AVISO] Debe detectarse exactamente una carta.')
                else:
                    guardar(idx, carpeta, nombre)
            elif tecla == ord('e'):
                if idx is None:
                    print('[AVISO] Sin carta valida para evidencia.')
                else:
                    guardar(vista, EVIDENCE / nombre / 'dealer', 'dealer')
                    guardar(corregida, EVIDENCE / nombre / 'cartas', 'carta')
                    guardar(idx, EVIDENCE / nombre / 'indices', 'indice')
            elif tecla == ord('a'):
                automatico = not automatico
                contador = 0
                ultima = time.monotonic()
                print('[OK] AUTO iniciado.' if automatico else '[OK] AUTO detenido.')
            elif tecla == ord('q'):
                break
    finally:
        camara.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
