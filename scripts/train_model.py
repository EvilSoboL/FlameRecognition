import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from multiprocessing import freeze_support
from tqdm import tqdm
from dataset import FlameDataset

# Параметры
BATCH_SIZE = 16
NUM_EPOCHS = 30
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Трансформы
train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(224, scale=(0.9,1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Функция MAPE
def mape_loss(preds, targets, eps=1e-6):
    return torch.mean(torch.abs((preds - targets) / (targets + eps)))

# Основная функция обучения
def main():
    # Пути к данным
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    train_csv = os.path.join(DATA_DIR, 'train.csv')
    val_csv   = os.path.join(DATA_DIR, 'val.csv')

    # Инициализация наборов данных и загрузчиков
    train_ds = FlameDataset(train_csv, transform=train_transform)
    val_ds   = FlameDataset(val_csv,   transform=eval_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Модель
    print(f"Using device: {DEVICE}")
    backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_feats = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Linear(num_feats, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
    model = backbone.to(DEVICE)

    # Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_mape = float('inf')
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        # ===== TRAIN =====
        model.train()
        train_losses = []
        train_bar = tqdm(train_loader, desc='  Training', unit='batch')
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs)
            loss_mse = criterion(preds, labels)
            loss_mape = mape_loss(preds, labels)
            loss = 0.5 * loss_mse + 0.5 * loss_mape
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # ===== VALIDATION =====
        model.eval()
        val_losses, val_maps = [], []
        val_bar = tqdm(val_loader, desc='  Validating', unit='batch')
        with torch.no_grad():
            for imgs, labels in val_bar:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs)
                loss_mse = criterion(preds, labels)
                loss_mape = mape_loss(preds, labels)
                loss = 0.5 * loss_mse + 0.5 * loss_mape
                val_losses.append(loss.item())
                val_maps.append(loss_mape.item())
                val_bar.set_postfix({'val_mape': f'{loss_mape.item()*100:.2f}%'})

        avg_train = sum(train_losses) / len(train_losses)
        avg_val   = sum(val_losses) / len(val_losses)
        avg_mape  = sum(val_maps) / len(val_maps)
        elapsed = time.time() - start_time

        print(f"Epoch {epoch} done in {elapsed:.1f}s - Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val MAPE: {avg_mape*100:.2f}%")

        scheduler.step(avg_val)

        # Checkpoint при улучшении MAPE
        if avg_mape < best_val_mape:
            best_val_mape = avg_mape
            ckpt_path = os.path.join(PROJECT_ROOT, 'best_model.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved best model to {ckpt_path} (Val MAPE: {avg_mape*100:.2f}%)")

    print("\nTraining complete.")

if __name__ == '__main__':
    freeze_support()
    main()