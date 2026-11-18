"""Evalúa el modelo inicial de BlackjackPrototype por carta y matriz de confusión."""
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / 'dataset' / 'training'
MODEL_PATH = ROOT / 'models' / 'blackjack_model.pth'
EXTENSIONS = {'.jpg', '.jpeg', '.png'}

class CardDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as img:
            return self.transform(img.convert('RGB')), label, str(path)

class SmallCNN(torch.nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        nn = torch.nn
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Dropout(0.25), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def test_samples(classes, seed):
    rng = random.Random(seed)
    samples = []
    for label, name in enumerate(classes):
        folder = DATASET / name
        if not folder.is_dir():
            raise FileNotFoundError(f'No existe la carpeta: {folder}')
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if len(files) < 10:
            raise ValueError(f'Faltan imágenes para {name}: {len(files)}')
        rng.shuffle(files)
        n_test = max(1, round(len(files) * 0.15))
        samples.extend((p, label) for p in files[:n_test])
    return samples

def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f'No se encontró el modelo: {MODEL_PATH}')
    # Solo cargamos un archivo generado por nuestro propio entrenamiento.
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    classes = checkpoint['classes']
    seed = checkpoint['seed']
    width, height = checkpoint['image_size']
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    samples = test_samples(classes, seed)
    loader = DataLoader(CardDataset(samples, transform), batch_size=16, shuffle=False, num_workers=0)
    model = SmallCNN(len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    confusion = [[0 for _ in classes] for _ in classes]
    mistakes = []
    with torch.inference_mode():
        for images, labels, paths in loader:
            predictions = model(images).argmax(dim=1).tolist()
            for actual, predicted, path in zip(labels.tolist(), predictions, paths):
                confusion[actual][predicted] += 1
                if actual != predicted:
                    mistakes.append((classes[actual], classes[predicted], path))

    print('\nRESULTADOS POR CARTA')
    print(f'{"Carta":<10} {"Aciertos":<12} {"Precisión":>10}')
    for i, name in enumerate(classes):
        total = sum(confusion[i]); correct = confusion[i][i]
        print(f'{name:<10} {f"{correct}/{total}":<12} {correct/total:>9.1%}')
    print('\nMATRIZ DE CONFUSIÓN (filas: real, columnas: predicción)')
    print(' ' * 10 + ''.join(f'{name:>7}' for name in classes))
    for name, row in zip(classes, confusion):
        print(f'{name:<10}' + ''.join(f'{n:>7}' for n in row))
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[i][i] for i in range(len(classes)))
    print(f'\nPRECISIÓN TOTAL: {correct}/{total} = {correct/total:.1%}')
    print('\nERRORES (real -> predicción):')
    if not mistakes:
        print('Ninguno.')
    else:
        for actual, predicted, path in mistakes:
            print(f'  {actual} -> {predicted} | {Path(path).name}')
    print('\nNota: se reconstruye la misma partición de prueba usando la semilla y el orden del entrenamiento original.')
    print('Si cambiaste las imágenes desde el entrenamiento, esta evaluación puede dejar de corresponder al conjunto original.')

if __name__ == '__main__':
    main()
