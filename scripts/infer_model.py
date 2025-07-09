import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Параметры (подстройте пути и Device при необходимости)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'best_model.pth')

# Трансформ для инференса (тот же, что для валидации)
eval_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Функция загрузки и предсказания одного изображения
def predict_image(model, image_path):
    img = Image.open(image_path).convert('RGB')
    inp = eval_transform(img).unsqueeze(0).to(DEVICE)  # shape [1,3,224,224]
    model.eval()
    with torch.no_grad():
        out = model(inp)
    # out: tensor([[fuel_flow, diluent_flow]])
    fuel_pred, dil_pred = out[0].cpu().numpy()
    return fuel_pred, dil_pred

# Функция выбора модели по имени файла
def create_model_from_path(path):
    filename = os.path.basename(path).lower()
    if 'densenet' in filename:
        backbone = models.densenet121(weights=None)
        num_feats = backbone.classifier.in_features
        backbone.classifier = nn.Sequential(
            nn.Linear(num_feats, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    else:
        backbone = models.efficientnet_b0(weights=None)
        num_feats = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Linear(num_feats, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    return backbone

# Основная функция
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for FlameRecognition model')
    parser.add_argument('images', nargs='+', help='Paths to input image files')
    parser.add_argument('--model', default=MODEL_PATH, help='Path to .pth model file')
    args = parser.parse_args()

    # Загрузка модели
    model = create_model_from_path(args.model).to(DEVICE)
    state = torch.load(args.model, map_location=DEVICE)
    model.load_state_dict(state)
    print(f'Model loaded from {args.model} on {DEVICE}')

    # Прогон инференса по списку изображений
    for img_path in args.images:
        if not os.path.isfile(img_path):
            print(f'File not found: {img_path}'); continue
        fuel, diluent = predict_image(model, img_path)
        print(f'Image: {img_path}')
        print(f'  Predicted fuel_flow:   {fuel:.3f} g/s')
        print(f'  Predicted diluent_flow: {diluent:.3f} g/s')
        print('-' * 40)

    print('Inference complete.')
