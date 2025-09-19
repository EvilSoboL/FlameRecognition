#!/usr/bin/env python3
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from multiprocessing import freeze_support
from tqdm import tqdm
from dataset import FlameDataset
import logging
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import argparse

# === Трансформы (основные + немного расширенные аугментации) ===
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Функция MAPE (безопасная) ===
def safe_mape(preds, targets, eps=1e-6):
    return torch.mean(torch.abs((preds - targets) / (targets.abs() + eps)))

# === EarlyStopping ===
class EarlyStopping:
    def __init__(self, patience=12, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0

    def step(self, value):
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return False
        else:
            self.bad_epochs += 1
            return self.bad_epochs > self.patience

# === Вспомогательные: вычисление mean/std по таргетам ===
def compute_target_stats(dataset: Dataset, device='cpu', num_workers=0):
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers)
    sums = None
    sq_sums = None
    n = 0
    for _, labels in tqdm(loader, desc="Computing target stats"):
        labels = labels.to(device)
        if sums is None:
            sums = labels.sum(dim=0)
            sq_sums = (labels ** 2).sum(dim=0)
        else:
            sums += labels.sum(dim=0)
            sq_sums += (labels ** 2).sum(dim=0)
        n += labels.shape[0]
    mean = (sums / n).cpu()
    var = (sq_sums / n - mean ** 2).clamp(min=0).cpu()
    std = torch.sqrt(var)
    std[std == 0] = 1.0
    return mean, std

# === Wrapper для нормализации таргетов внутри датасета ===
class NormTargetDataset(Dataset):
    def __init__(self, base_ds, mean, std):
        self.base = base_ds
        self.mean = torch.tensor(mean).float()
        self.std = torch.tensor(std).float()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        # предполагаем label — torch.Tensor shape (2,) или (N,)
        label = (label - self.mean) / self.std
        return img, label

# === Denormalize helper для метрик/инференса ===
class Denormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).float()
        self.std = torch.tensor(std).float()

    def __call__(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)

# === TTA: простой average original + horizontal flip ===
def tta_predict(model, imgs, device, tta_enabled=False):
    model.eval()
    with torch.no_grad():
        imgs = imgs.to(device)
        p1 = model(imgs)
        if not tta_enabled:
            return p1
        imgs_flip = torch.flip(imgs, dims=[3])
        p2 = model(imgs_flip)
        return 0.5 * (p1 + p2)

def main(args):
    freeze_support()
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")

    train_csv = os.path.join(DATA_DIR, f"{args.source}_train.csv")
    val_csv   = os.path.join(DATA_DIR, f"{args.source}_val.csv")

    if not os.path.isfile(train_csv) or not os.path.isfile(val_csv):
        raise FileNotFoundError(f"Не найдены {train_csv} или {val_csv}. Сначала сгенерируй датасет для '{args.source}'.")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(PROJECT_ROOT, "runs", f"run_{timestamp}_{args.source}")
    os.makedirs(run_dir, exist_ok=True)

    metrics_csv = os.path.join(run_dir, "training_metrics.csv")
    summary_txt = os.path.join(run_dir, "training_summary.txt")

    # CSV заголовок (добавлены per-output MAPE)
    with open(metrics_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_mape(%)", "val_mape_out1(%)", "val_mape_out2(%)", "elapsed(s)", "lr"])

    # === Датасеты и загрузчики ===
    train_ds_raw = FlameDataset(train_csv, transform=train_transform)
    val_ds_raw   = FlameDataset(val_csv, transform=eval_transform)

    # Вычислим mean/std по таргетам train
    print("Computing target mean/std on train set...")
    target_mean, target_std = compute_target_stats(FlameDataset(train_csv, transform=eval_transform), device='cpu', num_workers=0)
    print(f"Target mean: {target_mean.numpy()}, std: {target_std.numpy()}")

    # Оборачиваем датасеты для нормализации таргетов
    train_ds = NormTargetDataset(train_ds_raw, target_mean, target_std)
    val_ds   = NormTargetDataset(val_ds_raw, target_mean, target_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # === Сохраняем первое изображение после трансформов ===
    first_img, first_label = next(iter(train_loader))
    img_tensor = first_img[0].cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_denorm = img_tensor * std + mean
    img_denorm = img_denorm.clamp(0, 1)
    first_img_path = os.path.join(run_dir, "first_input.png")
    plt.imsave(first_img_path, img_denorm.permute(1, 2, 0).numpy())
    print(f"Saved first input image to {first_img_path}")

    # === Модель: DenseNet121 (с более мощной головой и Dropout) ===
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_feats = backbone.classifier.in_features
    backbone.classifier = nn.Sequential(
        nn.Linear(num_feats, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2)
    )
    model = backbone.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.SmoothL1Loss()  # робастная альтернатива MSE
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == 'onecycle':
        steps_per_epoch = max(1, len(train_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.max_lr, steps_per_epoch=steps_per_epoch,
            epochs=args.epochs, pct_start=0.1, anneal_strategy='cos', final_div_factor=100
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    best_val_mape = float('inf')
    earlystop = EarlyStopping(patience=args.early_stop_patience)

    logging.basicConfig(filename=os.path.join(run_dir, 'training.log'),
                        level=logging.INFO, filemode='a')
    logger = logging.getLogger()

    # Запись summary
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Training started\n")
        f.write(f"Model: DenseNet121\nDevice: {device}\n")
        f.write(f"Dataset: {args.source}\nBatch size: {args.batch_size}, Epochs: {args.epochs}, LR: {args.lr}\n")
        f.write(f"Target mean: {target_mean.numpy().tolist()}\nTarget std: {target_std.numpy().tolist()}\n\n")

    history = {"train_loss": [], "val_loss": [], "val_mape": []}
    denorm = Denormalize(target_mean, target_std)

    print(f"Starting training for {args.epochs} epochs on dataset '{args.source}'...")
    logging.info(f"Starting training for {args.epochs} epochs on dataset '{args.source}'...")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        logging.info(f"\nEpoch {epoch}/{args.epochs}")
        print(f"\nEpoch {epoch}/{args.epochs}")

        # --- TRAIN ---
        model.train()
        train_losses = []
        train_bar = tqdm(train_loader, desc='  Training', unit='batch')
        for imgs, labels in train_bar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)  # нормализованные метки
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                preds = model(imgs)  # в нормализованном пространстве
                loss_reg = criterion(preds, labels)
                if args.combine_mape:
                    preds_den = denorm(preds)
                    labels_den = denorm(labels)
                    loss_mape = safe_mape(preds_den, labels_den)
                    loss = args.alpha * loss_reg + args.beta * loss_mape
                else:
                    loss = loss_reg

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            if args.scheduler == 'onecycle':
                scheduler.step()

            train_losses.append(loss.item())
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train = sum(train_losses) / max(1, len(train_losses))

        # --- VALIDATION ---
        model.eval()
        val_losses = []
        val_maps = []
        val_maps_out1 = []
        val_maps_out2 = []
        val_bar = tqdm(val_loader, desc='  Validating', unit='batch')
        with torch.no_grad():
            for imgs, labels in val_bar:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                preds_norm = tta_predict(model, imgs, device, tta_enabled=args.tta)
                loss_reg = criterion(preds_norm, labels)
                preds_den = denorm(preds_norm)
                labels_den = denorm(labels)
                loss_mape = safe_mape(preds_den, labels_den)
                val_losses.append(loss_reg.item())
                val_maps.append(loss_mape.item())
                # per-output MAPE
                per_out = torch.mean(torch.abs((preds_den - labels_den) / (labels_den.abs().clamp(min=1e-6))), dim=0)
                val_maps_out1.append(per_out[0].item())
                val_maps_out2.append(per_out[1].item())
                val_bar.set_postfix({'val_mape': f'{loss_mape.item()*100:.2f}%'})

        avg_val = sum(val_losses) / max(1, len(val_losses))
        avg_mape = (sum(val_maps) / max(1, len(val_maps))) * 100.0
        avg_mape_out1 = (sum(val_maps_out1) / max(1, len(val_maps_out1))) * 100.0
        avg_mape_out2 = (sum(val_maps_out2) / max(1, len(val_maps_out2))) * 100.0
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_mape"].append(avg_mape)

        print(f"Epoch {epoch} done in {elapsed:.1f}s - Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val MAPE: {avg_mape:.2f}% (o1:{avg_mape_out1:.2f}% o2:{avg_mape_out2:.2f}%) | LR: {current_lr:.2e}")
        logging.info(f"Epoch {epoch} done in {elapsed:.1f}s - Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val MAPE: {avg_mape:.2f}% | LR: {current_lr:.2e}")

        # запись CSV
        with open(metrics_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train, avg_val, avg_mape, avg_mape_out1, avg_mape_out2, elapsed, current_lr])

        # запись summary
        with open(summary_txt, "a", encoding="utf-8") as f:
            f.write(f"Epoch {epoch}/{args.epochs} - Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}, Val MAPE: {avg_mape:.2f}%, Time: {elapsed:.1f}s, LR: {current_lr:.2e}\n")

        # scheduler step if plateau
        if args.scheduler != 'onecycle':
            scheduler.step(avg_val)

        # чекпоинт по лучшему val_mape
        if avg_mape < best_val_mape - 1e-9:
            best_val_mape = avg_mape
            model_filename = f'best_model_densenet_epoch{epoch}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
            ckpt_path = os.path.join(run_dir, model_filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'target_mean': target_mean,
                'target_std': target_std
            }, ckpt_path)
            logger.info(f"  Saved best DenseNet model to {ckpt_path} (Val MAPE: {avg_mape:.2f}%)")
            print(f"  Saved best DenseNet model to {ckpt_path} (Val MAPE: {avg_mape:.2f}%)")

        # early stopping
        if earlystop.step(avg_mape):
            print(f"Early stopping triggered (no improvement for {args.early_stop_patience} epochs).")
            break

    logging.info("\nTraining complete.")
    print("\nTraining complete.")
    with open(summary_txt, "a", encoding="utf-8") as f:
        f.write("\nTraining complete.\n")

    # === Графики ===
    try:
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
        plt.plot(history["val_mape"], label="Validation MAPE (%)")
        plt.xlabel("Epoch")
        plt.ylabel("MAPE (%)")
        plt.title("Validation MAPE")
        plt.legend()
        plt.savefig(os.path.join(run_dir, "val_mape.png"))
        plt.close()
    except Exception as e:
        print("Plotting failed:", e)

if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser(description="Train DenseNet121 on FlameDataset (modified)")
    parser.add_argument("--source", type=str, default="images",
                        help="Имя набора данных (например: 'images' или 'cropped_images')")
    parser.add_argument("--epochs", type=int, default=70, help="Количество эпох обучения")
    parser.add_argument("--batch_size", type=int, default=16, help="Размер батча")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate (used for AdamW)")
    parser.add_argument("--max_lr", type=float, default=1e-3, help="Max LR for OneCycle (if selected)")
    parser.add_argument("--scheduler", type=str, choices=['onecycle', 'plateau'], default='plateau', help="Scheduler to use")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay for optimizer")
    parser.add_argument("--combine_mape", action='store_true', help="Combine SmoothL1 loss with MAPE on denormalized outputs")
    parser.add_argument("--alpha", type=float, default=1.0, help="weight for reg loss when combining")
    parser.add_argument("--beta", type=float, default=0.1, help="weight for mape when combining")
    parser.add_argument("--tta", action='store_true', help="Enable simple TTA (horizontal flip) during validation")
    parser.add_argument("--early_stop_patience", type=int, default=12, help="Early stopping patience (epochs)")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm (0 to disable)")
    args = parser.parse_args()
    main(args)
