import os
import argparse
from PIL import Image
from tqdm import tqdm

class ManualCrop:
    """Кастомный трансформ: ручной кроп с масштабом и смещением"""
    def __init__(self, scale=0.6, offset_x=200, offset_y=-300):
        self.scale = scale
        self.offset_x = offset_x
        self.offset_y = offset_y

    def __call__(self, img: Image.Image):
        w, h = img.size
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)
        center_x = w // 2 + self.offset_x
        center_y = h // 2 + self.offset_y
        left   = max(0, center_x - new_w // 2)
        top    = max(0, center_y - new_h // 2)
        right  = min(w, center_x + new_w // 2)
        bottom = min(h, center_y + new_h // 2)
        return img.crop((left, top, right, bottom))


def process_images(input_root: str, output_root: str, cropper: ManualCrop):
    """
    input_root  - папка с оригинальными изображениями
    output_root - папка для сохранения результата
    """
    os.makedirs(output_root, exist_ok=True)

    for subdir, _, files in os.walk(input_root):
        # относительный путь к подпапке
        rel_path = os.path.relpath(subdir, input_root)
        output_subdir = os.path.join(output_root, rel_path)
        os.makedirs(output_subdir, exist_ok=True)

        for file in tqdm(files, desc=f"Processing {rel_path}"):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                input_path = os.path.join(subdir, file)
                output_path = os.path.join(output_subdir, file)

                try:
                    with Image.open(input_path) as img:
                        cropped = cropper(img)
                        cropped.save(output_path)
                except Exception as e:
                    print(f"❌ Ошибка при обработке {input_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch crop images with ManualCrop")
    parser.add_argument("--input", type=str, default="images", help="Папка с исходными изображениями")
    parser.add_argument("--output", type=str, default="cropped_images", help="Папка для сохранения обрезанных изображений")
    parser.add_argument("--scale", type=float, default=0.6, help="Масштаб кропа (0..1)")
    parser.add_argument("--offset_x", type=int, default=200, help="Смещение по X (пиксели)")
    parser.add_argument("--offset_y", type=int, default=-300, help="Смещение по Y (пиксели)")
    args = parser.parse_args()

    cropper = ManualCrop(scale=args.scale, offset_x=args.offset_x, offset_y=args.offset_y)
    process_images(args.input, args.output, cropper)

    print(f"✅ Все изображения обработаны и сохранены в {args.output}")
