import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Путь к CSV и изображениям
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
IMG_DIR = os.path.join(DATA_DIR, 'images')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')

# Преобразование изображения
train_transform = transforms.Compose([
    transforms.Resize((256,256)),  # Стандартизация размеров

    # Случайно вырезает область изображения (224×224) с масштабом от 90% до 100% исходного размера. Это:
    # - Увеличивает разнообразие тренировочных данных.
    # - Помогает модели стать устойчивой к изменениям положения объектов, масштаба и ракурса.
    # - Снижает риск переобучения (overfitting).
    transforms.RandomResizedCrop(224, scale=(0.9,1.0)),

    # Конвертирует изображение (формат PIL или numpy) в тензор PyTorch с размерностью [C, H, W] и автоматически нормирует значения пикселей в диапазон [0, 1].
    transforms.ToTensor(),

    # Стандартизует значения тензора по каналам (используемые значения mean и std соответствуют статистике датасета ImageNet.
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


# Dataset
class FlameDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(PROJECT_ROOT, row.image_path)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor([row.fuel_flow, row.diluent_flow], dtype=torch.float32)
        return img, label

# Проверка загрузчика
def check_loader():
    ds = FlameDataset(TRAIN_CSV, transform=train_transform)
    dl = DataLoader(ds, batch_size=8, shuffle=True)

    # Берём один батч
    imgs, labels = next(iter(dl))

    # Форма тензоров
    print(f"Batch images shape: {imgs.shape}")  # [B, C, H, W]
    print(f"Labels shape: {labels.shape}")        # [B, 2]

    # Статистика меток
    fuel = labels[:,0]
    dilu = labels[:,1]
    print(f"fuel_flow min/max: {fuel.min().item()}/{fuel.max().item()}")
    print(f"diluent_flow min/max: {dilu.min().item()}/{dilu.max().item()}")

    # Визуализация нескольких изображений с подписями
    inv_norm = transforms.Normalize(
        mean=[-m/s for m, s in zip([0.485,0.456,0.406],[0.229,0.224,0.225])],
        std=[1/s for s in [0.229,0.224,0.225]]
    )
    plt.figure(figsize=(12,6))
    for i in range(min(4, imgs.size(0))):
        ax = plt.subplot(1,4,i+1)
        img = inv_norm(imgs[i]).permute(1,2,0).clamp(0,1).numpy()
        ax.imshow(img)
        ax.set_title(f"Fuel: {fuel[i].item():.2f}\nDiluent: {dilu[i].item():.2f}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    check_loader()
