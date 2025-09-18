import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from multiprocessing import freeze_support
from tqdm import tqdm
from dataset import FlameDataset
import logging
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

# Параметры
BATCH_SIZE = 16
NUM_EPOCHS = 70
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# === Обновленные трансформы ===
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

    # === Папка для результатов ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(PROJECT_ROOT, "runs", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    metrics_csv = os.path.join(run_dir, "training_metrics.csv")
    summary_txt = os.path.join(run_dir, "training_summary.txt")

    # Создаем CSV и записываем заголовки
    with open(metrics_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_mape(%)", "elapsed(s)", "lr"])

    # Датасеты и загрузчики
    train_ds = FlameDataset(train_csv, transform=train_transform)
    val_ds   = FlameDataset(val_csv,   transform=eval_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Датасеты и загрузчики
    train_ds = FlameDataset(train_csv, transform=train_transform)
    val_ds   = FlameDataset(val_csv,   transform=eval_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # === Сохраняем первое изображение после трансформов ===
    first_img, first_label = next(iter(train_loader))  # берём первую партию
    img_tensor = first_img[0].cpu()  # первое изображение из батча
    # Обратно денормализуем
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img_denorm = img_tensor * std + mean
    img_denorm = img_denorm.clamp(0,1)  # ограничиваем значения
    # Сохраняем
    plt.imsave(os.path.join(run_dir, "first_input.png"), img_denorm.permute(1,2,0).numpy())
    print(f"Saved first input image to {os.path.join(run_dir, 'first_input.png')}")


    # Модель: DenseNet121
    print(f"Using device: {DEVICE}")
    backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_feats = backbone.classifier.in_features
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
    print(f"Starting training for {NUM_EPOCHS} epochs with DenseNet121...")

    logging.basicConfig(filename=os.path.join(run_dir, 'training.log'), level=logging.INFO, filemode='a')
    logger = logging.getLogger()

    # Запишем параметры обучения в summary.txt
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Training started\n")
        f.write(f"Model: DenseNet121\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}, LR: {LR}\n\n")

    # Для графиков
    history = {"train_loss": [], "val_loss": [], "val_mape": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        logging.info(f"\nEpoch {epoch}/{NUM_EPOCHS}")
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
        current_lr = optimizer.param_groups[0]['lr']

        # Сохраним в историю
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_mape"].append(avg_mape * 100)

        print(
            f"Epoch {epoch} done in {elapsed:.1f}s - Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val MAPE: {avg_mape * 100:.2f}% | LR: {current_lr:.2e}"
        )
        logging.info(
            f"Epoch {epoch} done in {elapsed:.1f}s - Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val MAPE: {avg_mape * 100:.2f}% | LR: {current_lr:.2e}"
        )

        # === Запись в CSV ===
        with open(metrics_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train, avg_val, avg_mape * 100, elapsed, current_lr])

        # === Запись в TXT (summary) ===
        with open(summary_txt, "a", encoding="utf-8") as f:
            f.write(
                f"Epoch {epoch}/{NUM_EPOCHS} - "
                f"Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}, "
                f"Val MAPE: {avg_mape*100:.2f}%, Time: {elapsed:.1f}s, LR: {current_lr:.2e}\n"
            )

        scheduler.step(avg_val)

        # Checkpoint при улучшении MAPE
        if avg_mape < best_val_mape:
            best_val_mape = avg_mape
            # Добавляем timestamp к имени модели
            model_filename = f'best_model_densenet_epoch{epoch}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
            ckpt_path = os.path.join(run_dir, model_filename)
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  Saved best DenseNet model to {ckpt_path} (Val MAPE: {avg_mape * 100:.2f}%)")
            print(f"  Saved best DenseNet model to {ckpt_path} (Val MAPE: {avg_mape * 100:.2f}%)")

    logging.info("\nTraining complete.")
    print("\nTraining complete.")
    with open(summary_txt, "a", encoding="utf-8") as f:
        f.write("\nTraining complete.\n")

    # === Построение графиков ===
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss curves")
    plt.savefig(os.path.join(run_dir, "loss_curves.png"))
    plt.close()

    plt.figure()
    plt.plot(history["val_mape"], label="Validation MAPE (%)", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("MAPE (%)")
    plt.title("Validation MAPE")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "val_mape.png"))
    plt.close()

if __name__ == '__main__':
    freeze_support()
    main()
