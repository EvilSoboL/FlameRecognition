import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Определяем корень проекта относительно расположения этого скрипта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

# Параметры путей (корень проекта FlameRecognition)
DATA_ROOT   = os.path.join(PROJECT_ROOT, "data", "images")   # <project>/data/images
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "data")
FULL_CSV    = os.path.join(OUTPUT_DIR, "data.csv")

# 1. Собираем записи из подпапок data/images/<steam>_<diesel>
records = []
if not os.path.isdir(DATA_ROOT):
    raise FileNotFoundError(f"DATA_ROOT not found: {DATA_ROOT}")

for mode in sorted(os.listdir(DATA_ROOT)):
    mode_dir = os.path.join(DATA_ROOT, mode)
    if not os.path.isdir(mode_dir):
        continue
    # Парсим расход пара (diluent) и расход диз. топлива (fuel)
    try:
        steam_str, diesel_str = mode.split("_")
        diluent_flow = float(steam_str)   # вводимый пар (г/с)
        fuel_flow   = float(diesel_str)   # дизельное топливо (г/с)
    except Exception:
        print(f"WARNING: папка '{mode}' не соответствует шаблону '<steam>_<diesel>' и будет пропущена")
        continue

    # Проходим по файлам изображений
    for fname in sorted(os.listdir(mode_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join("data", "images", mode, fname)  # относительный путь от project_root
        records.append({
            'image_path': img_path,
            'fuel_flow': fuel_flow,
            'diluent_flow': diluent_flow
        })

# 2. Сохраняем полный CSV
os.makedirs(OUTPUT_DIR, exist_ok=True)
full_df = pd.DataFrame(records)
full_df.to_csv(FULL_CSV, index=False)
print(f"Saved total {len(full_df)} records to {FULL_CSV}")

# 3. Разбиваем на train/val/test (80/10/10)
train_df, tmp_df = train_test_split(full_df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=42)

# Добавляем столбец split
train_df['split'] = 'train'
val_df['split']   = 'val'
test_df['split']  = 'test'

# 4. Сохраняем разбиение
train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'),   index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
print(f"Split counts -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

# 5. Пример записи
print(train_df.head())
