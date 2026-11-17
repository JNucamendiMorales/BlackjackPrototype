"""Revisa y limpia copias de índices de Blackjack, sin tocar los originales.
Ejecutar desde la raíz de BlackjackPrototype:
    python clean_blackjack_indices.py
"""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'dataset' / 'indices'
DEST = ROOT / 'dataset' / 'indices_clean'


def clean_image(img):
    h, w = img.shape[:2]
    # Solo considerar fondo negro real si hay grandes márgenes negros.
    # Las cartas oscuras no se recortan: se busca contenido claramente no negro.
    if img.ndim == 2:
        nonblack = img > 22
    else:
        nonblack = np.max(img[:, :, :3], axis=2) > 22
    ys, xs = np.where(nonblack)
    if not len(xs):
        return img, 'sin contenido detectable'
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    box_area = (x2-x1)*(y2-y1)
    # No recortar si el contenido ocupa buena parte de la imagen.
    if box_area >= 0.65 * h * w:
        return img, 'ya ajustada'
    # Evitar recortes accidentales de imágenes oscuras o con ruido.
    if box_area < 0.001 * h * w or (x2-x1) < 8 or (y2-y1) < 12:
        return img, 'revisar manualmente'
    pad = 3
    x1, y1 = max(0, x1-pad), max(0, y1-pad)
    x2, y2 = min(w, x2+pad), min(h, y2+pad)
    return img[y1:y2, x1:x2].copy(), 'márgenes recortados'


def main():
    if not SOURCE.is_dir():
        raise SystemExit(f'[ERROR] No existe {SOURCE}')
    count = 0
    for folder in ('player_1', 'dealer'):
        src = SOURCE / folder
        if not src.exists():
            continue
        dst = DEST / folder
        dst.mkdir(parents=True, exist_ok=True)
        for path in sorted(src.glob('*.png')):
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f'[ERROR] No se pudo abrir {path}')
                continue
            fixed, status = clean_image(img)
            target = dst / path.name
            if not cv2.imwrite(str(target), fixed):
                print(f'[ERROR] No se pudo guardar {target}')
                continue
            print(f'{folder}/{path.name}: {img.shape[1]}x{img.shape[0]} -> '
                  f'{fixed.shape[1]}x{fixed.shape[0]} ({status})')
            count += 1
    print(f'\n[OK] {count} archivos revisados. Copias en: {DEST}')
    print('[INFO] Los originales no fueron modificados.')
    print('[IMPORTANTE] Revisa visualmente las copias antes de etiquetar o entrenar.')


if __name__ == '__main__':
    main()
