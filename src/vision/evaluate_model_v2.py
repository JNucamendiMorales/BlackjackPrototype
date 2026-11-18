"""Evalúa el modelo V2 de BlackjackPrototype sobre el conjunto de prueba original."""
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from train_model_v2 import CardCNNv2, CardDataset, CLASSES, DATASET, EXTENSIONS, SEED

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / 'models' / 'blackjack_model_v2.pth'


def original_test_samples():
    """Reproduce exactamente el orden y el muestreo de train_model_v2.py."""
    rng = random.Random(SEED)
    test = []
    for label, name in enumerate(CLASSES):
        folder = DATASET / name
        if not folder.is_dir():
            raise FileNotFoundError(f'No existe: {folder}')
        files = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if len(files) < 10:
            raise ValueError(f'La clase {name} tiene menos de 10 imágenes')
        rng.shuffle(files)
        n_test = max(1, round(len(files) * 0.15))
        test.extend((p, label) for p in files[:n_test])
    return test


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f'No se encontró el modelo: {MODEL_PATH}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    classes = checkpoint['classes']
    if classes != CLASSES:
        raise ValueError('Las clases guardadas no coinciden con train_model_v2.py')
    if checkpoint.get('architecture') != 'CardCNNv2':
        raise ValueError('El archivo no contiene la arquitectura CardCNNv2')
    width, height = checkpoint['image_size']
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    samples = original_test_samples()
    loader = DataLoader(CardDataset(samples, transform), batch_size=16,
                        shuffle=False, num_workers=0)
    model = CardCNNv2(len(classes)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    matrix = [[0 for _ in classes] for _ in classes]
    predictions = []
    with torch.inference_mode():
        for images, _ in loader:
            predictions.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())

    errors = []
    for (path, real), predicted in zip(samples, predictions):
        matrix[real][predicted] += 1
        if real != predicted:
            errors.append((classes[real], classes[predicted], path.name))

    print('\nRESULTADOS POR CARTA')
    print(f'{"Carta":<10} {"Aciertos":<14} Precisión')
    for i, name in enumerate(classes):
        total = sum(matrix[i])
        correct = matrix[i][i]
        print(f'{name:<10} {f"{correct}/{total}":<14} {correct / total:>7.1%}')

    col_width = max(7, max(map(len, classes)) + 2)
    print('\nMATRIZ DE CONFUSIÓN (filas: real, columnas: predicción)')
    print(' ' * 12 + ''.join(f'{name:>{col_width}}' for name in classes))
    for name, row in zip(classes, matrix):
        print(f'{name:<12}' + ''.join(f'{value:>{col_width}}' for value in row))

    total = sum(map(sum, matrix))
    correct = sum(matrix[i][i] for i in range(len(classes)))
    print(f'\nPRECISIÓN TOTAL: {correct}/{total} = {correct / total:.1%}')
    print('\nERRORES (real -> predicción):')
    if not errors:
        print('  Ninguno')
    for real, predicted, filename in errors:
        print(f'  {real} -> {predicted} | {filename}')
    if 'test_accuracy' in checkpoint:
        expected = checkpoint['test_accuracy']
        if abs(correct / total - expected) > 1e-6:
            print('\nADVERTENCIA: La precisión no coincide con la guardada en el modelo.')
            print('Revisa si agregaste, eliminaste o renombraste imágenes después del entrenamiento.')
    print('\nNota: la partición se reconstruye usando la semilla y los nombres originales.')


if __name__ == '__main__':
    main()
