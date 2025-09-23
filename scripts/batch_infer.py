#!/usr/bin/env python3
import os
import csv
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
from infer_model import robust_load_checkpoint, strip_module_prefix, create_model_matching_checkpoint, predict_image, denormalize


def parse_expected(folder_name: str):
    """
    Парсим эталонные значения из имени папки.
    Пример: S5D900 -> diluent=500, fuel=900
    """
    match = re.match(r"S(\d+)D(\d+)", folder_name, re.IGNORECASE)
    if not match:
        return None, None
    s_val = int(match.group(1)) * 100
    d_val = int(match.group(2))
    return s_val, d_val


def infer_folder(image_dir, model, target_mean, target_std, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    folder_name = os.path.basename(image_dir)
    expected_diluent, expected_fuel = parse_expected(folder_name)

    exts = {".jpg", ".jpeg", ".png"}
    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in exts]
    images.sort()

    results = []
    for img_path in images:
        raw_pred = predict_image(model, img_path, device)
        if target_mean is not None and target_std is not None:
            den = denormalize(raw_pred, target_mean, target_std)
            fuel, diluent = den[0], den[1]
        else:
            fuel, diluent = raw_pred[0], raw_pred[1]
        results.append((os.path.basename(img_path), fuel, diluent, expected_fuel, expected_diluent))

    # Сохраняем в CSV
    csv_path = os.path.join(save_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "predicted_fuel_flow", "predicted_diluent_flow",
                         "expected_fuel_flow", "expected_diluent_flow"])
        writer.writerows(results)

    # Локальный график
    fuels = [r[1] for r in results]
    diluents = [r[2] for r in results]
    x = np.arange(len(results))

    plt.figure(figsize=(10, 5))
    plt.plot(x, fuels, label="Predicted Fuel", marker="o")
    plt.plot(x, diluents, label="Predicted Diluent", marker="s")
    if expected_fuel is not None and expected_diluent is not None:
        plt.axhline(expected_fuel, color="red", linestyle="--", label=f"Expected Fuel = {expected_fuel}")
        plt.axhline(expected_diluent, color="green", linestyle="--", label=f"Expected Diluent = {expected_diluent}")
    plt.xlabel("Image index")
    plt.ylabel("Flow value")
    plt.title(f"Predicted flows for {folder_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "plot.png")
    plt.savefig(plot_path)
    plt.close()

    return results, csv_path, plot_path


def main(args):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Загружаем модель
    state_dict_raw, extras = robust_load_checkpoint(args.model, map_location=device)
    state_dict = strip_module_prefix(state_dict_raw)
    model = create_model_matching_checkpoint(state_dict, device=device).to(device)
    model.load_state_dict(state_dict, strict=False)

    target_mean = extras.get("target_mean")
    target_std = extras.get("target_std")
    if isinstance(target_mean, torch.Tensor): target_mean = target_mean.cpu().numpy()
    if isinstance(target_std, torch.Tensor): target_std = target_std.cpu().numpy()

    base_dir = args.images
    result_dir = args.output
    os.makedirs(result_dir, exist_ok=True)

    all_results = []
    for folder in sorted(os.listdir(base_dir)):
        if folder.lower().startswith("s"):  # только папки, начинающиеся на S
            img_dir = os.path.join(base_dir, folder)
            if not os.path.isdir(img_dir):
                continue
            save_dir = os.path.join(result_dir, folder)
            results, csv_path, plot_path = infer_folder(img_dir, model, target_mean, target_std, device, save_dir)
            for r in results:
                all_results.append((folder, r[0], r[1], r[2], r[3], r[4]))
            print(f"Processed {folder}: saved {csv_path}, {plot_path}")

    # Сохраняем общий CSV
    summary_csv = os.path.join(result_dir, "all_results.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "image", "predicted_fuel_flow", "predicted_diluent_flow",
                         "expected_fuel_flow", "expected_diluent_flow"])
        writer.writerows(all_results)

    print(f"\nSaved summary CSV to {summary_csv}")

    # === Общие графики ===
    if all_results:
        # Fuel
        folders = [f"{r[0]}/{r[1]}" for r in all_results]
        pred_fuel = [r[2] for r in all_results]
        exp_fuel = [r[4] for r in all_results]

        plt.figure(figsize=(12, 6))
        plt.plot(pred_fuel, "bo-", label="Predicted Fuel")
        plt.plot(exp_fuel, "r--", label="Expected Fuel")
        plt.xlabel("Image index (all folders)")
        plt.ylabel("Fuel flow")
        plt.title("Summary Fuel Flow Predictions")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fuel_path = os.path.join(result_dir, "summary_fuel.png")
        plt.savefig(fuel_path)
        plt.close()
        print(f"Saved {fuel_path}")

        # Diluent
        pred_diluent = [r[3] for r in all_results]
        exp_diluent = [r[5] for r in all_results]

        plt.figure(figsize=(12, 6))
        plt.plot(pred_diluent, "go-", label="Predicted Diluent")
        plt.plot(exp_diluent, "m--", label="Expected Diluent")
        plt.xlabel("Image index (all folders)")
        plt.ylabel("Diluent flow")
        plt.title("Summary Diluent Flow Predictions")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        diluent_path = os.path.join(result_dir, "summary_diluent.png")
        plt.savefig(diluent_path)
        plt.close()
        print(f"Saved {diluent_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference for all S* folders in images/")
    parser.add_argument("--images", type=str, default="data/images", help="Base images directory")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output", type=str, default="result", help="Directory to save results")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    main(args)
