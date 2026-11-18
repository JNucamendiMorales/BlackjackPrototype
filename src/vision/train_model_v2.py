"""Segundo entrenamiento de BlackjackPrototype: CNN mejorada y comparación justa con v1."""
from pathlib import Path
import copy
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / 'dataset' / 'training'
OUTPUT = ROOT / 'models'
CLASSES = ['4_C', '7_T', '8_T', '9_D', '9_T', 'A_P', 'J_C', 'K_D']
EXTENSIONS = {'.jpg', '.jpeg', '.png'}
SEED = 42
IMAGE_SIZE = 128
EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 16
LR = 0.0005


class CardDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as img:
            return self.transform(img.convert('RGB')), label


class CardCNNv2(nn.Module):
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
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def split_samples():
    # MISMA semilla, orden, redondeo y distribución de v1.
    rng = random.Random(SEED)
    groups = {'train': [], 'val': [], 'test': []}
    for label, name in enumerate(CLASSES):
        folder = DATASET / name
        if not folder.is_dir():
            raise FileNotFoundError(f'Falta la carpeta: {folder}')
        files = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if len(files) < 10:
            raise ValueError(f'{name} tiene muy pocas imágenes: {len(files)}')
        rng.shuffle(files)
        n_test = max(1, round(len(files) * 0.15))
        n_val = max(1, round(len(files) * 0.15))
        groups['test'].extend((p, label) for p in files[:n_test])
        groups['val'].extend((p, label) for p in files[n_test:n_test + n_val])
        groups['train'].extend((p, label) for p in files[n_test + n_val:])
        print(f'{name}: {len(files)} imágenes')
    return groups


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Dispositivo: {device}')
    groups = split_samples()
    print('División:', {k: len(v) for k, v in groups.items()})

    base = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomAffine(degrees=8, translate=(0.05, 0.05),
                                scale=(0.95, 1.05), fill=255),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    loaders = {
        name: DataLoader(CardDataset(samples, train_transform if name == 'train' else base),
                         batch_size=BATCH_SIZE, shuffle=(name == 'train'), num_workers=0)
        for name, samples in groups.items()
    }

    model = CardCNNv2(len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0001)
    best_acc, best_loss, best_epoch = -1.0, float('inf'), 0
    best_state = None
    stale = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for images, labels in loaders['train']:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        _, train_acc = evaluate(model, loaders['train'], criterion, device)
        val_loss, val_acc = evaluate(model, loaders['val'], criterion, device)
        print(f'Época {epoch:02d}/{EPOCHS} | train {train_acc:.1%} | '
              f'validación {val_acc:.1%} | pérdida val {val_loss:.3f}', flush=True)

        # Ante igual precisión, preferir menor pérdida de validación.
        improved = val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss - 1e-5)
        if improved:
            best_acc, best_loss, best_epoch = val_acc, val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f'Detención anticipada: {PATIENCE} épocas sin mejora.')
                break

    model.load_state_dict(best_state)
    _, test_acc = evaluate(model, loaders['test'], criterion, device)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / 'blackjack_model_v2.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': 'CardCNNv2',
        'classes': CLASSES,
        'image_size': [IMAGE_SIZE, IMAGE_SIZE],
        'seed': SEED,
        'best_epoch': best_epoch,
        'validation_accuracy': best_acc,
        'test_accuracy': test_acc,
    }, path)
    print(f'\nMejor época: {best_epoch}')
    print(f'Mejor validación: {best_acc:.1%}')
    print(f'Prueba final: {test_acc:.1%}')
    print(f'Modelo guardado en: {path}')
    print('Nota: esta comparación solo es válida si el dataset no cambió desde v1.')


if __name__ == '__main__':
    main()
