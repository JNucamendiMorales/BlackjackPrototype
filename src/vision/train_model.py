"""Entrenamiento inicial de 8 cartas para BlackjackPrototype (CPU o CUDA)."""
from pathlib import Path
import random
import json
import copy

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / 'dataset' / 'training'
OUTPUT = ROOT / 'models'
CLASSES = ['4_C', '7_T', '8_T', '9_D', '9_T', 'A_P', 'J_C', 'K_D']
EXTENSIONS = {'.jpg', '.jpeg', '.png'}
SEED = 42
EPOCHS = 25
BATCH_SIZE = 16
LR = 0.001

class CardDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as img:
            img = img.convert('RGB')
            image = self.transform(img)
        return image, label

class SmallCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
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

def split_samples():
    rng = random.Random(SEED)
    groups = {'train': [], 'val': [], 'test': []}
    for label, name in enumerate(CLASSES):
        folder = DATASET / name
        if not folder.is_dir():
            raise FileNotFoundError(f'Falta la carpeta: {folder}')
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if len(files) < 10:
            raise ValueError(f'{name} tiene muy pocas imágenes: {len(files)}')
        rng.shuffle(files)
        n_test = max(1, round(len(files) * 0.15))
        n_val = max(1, round(len(files) * 0.15))
        for p in files[:n_test]:
            groups['test'].append((p, label))
        for p in files[n_test:n_test + n_val]:
            groups['val'].append((p, label))
        for p in files[n_test + n_val:]:
            groups['train'].append((p, label))
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
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Dispositivo: {device}')
    groups = split_samples()
    print('División:', {name: len(samples) for name, samples in groups.items()})

    base = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    loaders = {
        name: DataLoader(CardDataset(samples, train_transform if name == 'train' else base),
                         batch_size=BATCH_SIZE, shuffle=(name == 'train'), num_workers=0)
        for name, samples in groups.items()
    }
    model = SmallCNN(len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_accuracy = -1.0
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for images, labels in loaders['train']:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        train_loss, train_acc = evaluate(model, loaders['train'], criterion, device)
        val_loss, val_acc = evaluate(model, loaders['val'], criterion, device)
        print(f'Época {epoch:02d}/{EPOCHS} | train {train_acc:.1%} | validación {val_acc:.1%} | pérdida val {val_loss:.3f}', flush=True)
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, loaders['test'], criterion, device)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT / 'blackjack_model.pth'
    torch.save({'model_state_dict': model.state_dict(), 'classes': CLASSES,
                'image_size': [96, 96], 'seed': SEED,
                'validation_accuracy': best_accuracy, 'test_accuracy': test_acc}, model_path)
    print(f'\nMejor validación: {best_accuracy:.1%}')
    print(f'Prueba final: {test_acc:.1%}')
    print(f'Modelo guardado en: {model_path}')
    print('Nota: fotografías similares de una misma sesión pueden sobreestimar la precisión real.')

if __name__ == '__main__':
    main()
