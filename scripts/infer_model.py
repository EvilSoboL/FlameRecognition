#!/usr/bin/env python3
"""
Robust infer script that auto-reconstructs the classifier head from checkpoint keys
so that loading state_dict succeeds even if head size differs from local defaults.

Output clarification:
 - First number in prediction = predicted_fuel_flow
 - Second number in prediction = predicted_diluent_flow
"""

import os
import argparse
import re
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Inference transforms (same as validation)
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# -------------------------
# Helpers for checkpoint handling
# -------------------------
def robust_load_checkpoint(path, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict):
        for key in ('model_state_dict', 'state_dict', 'model_state', 'model'):
            if key in ckpt:
                return ckpt[key], {k: v for k, v in ckpt.items() if k != key}
        # if dict looks like a state_dict (layer keys)
        sample_keys = list(ckpt.keys())[:10]
        if any(isinstance(k, str) and (k.startswith('features.') or k.startswith('classifier.') or k.startswith('module.')) for k in sample_keys):
            return ckpt, {}
        # fallback: return whole thing as state candidate; caller will fail with clear message
        return ckpt, {}
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")

def strip_module_prefix(state_dict):
    new = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new[k[len('module.'):]] = v
        else:
            new[k] = v
    return new

# -------------------------
# Reconstruct classifier from state_dict keys
# -------------------------
def build_classifier_from_state(state_dict, backbone_out_features, device='cpu'):
    """
    Given state_dict and backbone_out_features (num features before classifier),
    reconstruct nn.Sequential for backbone.classifier with modules placed at the
    same numeric indices found in keys like 'classifier.0.weight', 'classifier.3.weight', etc.

    For indices that don't have parameters (e.g. ReLU/Dropout in original), we insert nn.ReLU()
    as a safe parameterless placeholder.
    """
    # find keys matching classifier.<idx>.weight
    pattern = re.compile(r'^classifier\.(\d+)\.weight$')
    idx_to_weightshape = {}
    max_idx = -1
    for k, v in state_dict.items():
        m = pattern.match(k)
        if m:
            idx = int(m.group(1))
            max_idx = max(max_idx, idx)
            # v is Tensor; shape [out_features, in_features]
            idx_to_weightshape[idx] = tuple(v.shape)

    if len(idx_to_weightshape) == 0:
        # No classifier weights in state_dict -> return a default small head
        print("Warning: no classifier.*.weight keys found in checkpoint; building default head.")
        return nn.Sequential(
            nn.Linear(backbone_out_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    # Build module list of length max_idx+1
    modules = OrderedDict()
    for i in range(max_idx + 1):
        key = f"{i}"
        if i in idx_to_weightshape:
            out_f, in_f = idx_to_weightshape[i]
            # some checkpoints store first linear as [512, 1024], where in_f==backbone_out_features
            # Create Linear(in_f -> out_f)
            modules[key] = nn.Linear(in_f, out_f)
        else:
            # placeholder parameterless module (ReLU)
            modules[key] = nn.ReLU()

    # The Sequential must be ordered with numeric indices as keys '0','1',...
    # But nn.Sequential wants modules in order; create a list preserving indices
    seq = nn.Sequential(OrderedDict([(str(i), modules[str(i)]) for i in range(max_idx + 1)]))
    return seq

# -------------------------
# Build backbone and attach reconstructed classifier
# -------------------------
def create_model_matching_checkpoint(state_dict, device='cpu'):
    """
    Build a DenseNet121 backbone and reconstruct classifier matching checkpoint keys/shapes.
    """
    # create backbone with pretrained weights disabled (we will load checkpoint weights)
    backbone = models.densenet121(weights=None)
    # get the original backbone output feat dim
    try:
        backbone_out = backbone.classifier.in_features
    except Exception:
        # fallback common value for DenseNet121
        backbone_out = 1024

    # reconstruct classifier by probing state_dict
    classifier = build_classifier_from_state(state_dict, backbone_out, device=device)

    # attach
    backbone.classifier = classifier
    return backbone

# -------------------------
# Denormalize helper
# -------------------------
def denormalize(preds, mean, std):
    mean = np.array(mean, dtype=float)
    std = np.array(std, dtype=float)
    return preds * std + mean

# -------------------------
# Predict single image
# -------------------------
def predict_image(model, img_path, device):
    img = Image.open(img_path).convert('RGB')
    inp = eval_transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(inp)
    return out.cpu().numpy().squeeze()

# -------------------------
# Main entry
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Robust infer - auto-match classifier to checkpoint")
    parser.add_argument("images", nargs='+', help="Paths to image files")
    parser.add_argument("--model", "-m", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--device", "-d", choices=['cpu','cuda'], default=None, help="Device (auto by default)")
    parser.add_argument("--no_denorm", action='store_true', help="Don't attempt to denormalize outputs")
    parser.add_argument("--verbose", action='store_true', help="Verbose prints")
    args = parser.parse_args()

    device = args.device if args.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = 'cpu'
    print(f"Using device: {device}")

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    # load checkpoint
    state_dict_raw, extras = robust_load_checkpoint(args.model, map_location=device)
    # strip module prefix if present
    state_dict = strip_module_prefix(state_dict_raw)

    # Build model matching the classifier layout in checkpoint
    model = create_model_matching_checkpoint(state_dict, device=device)
    model = model.to(device)

    # Try to load state_dict
    try:
        load_res = model.load_state_dict(state_dict, strict=False)
        missing = getattr(load_res, 'missing_keys', None) or (load_res.get('missing_keys') if isinstance(load_res, dict) else None)
        unexpected = getattr(load_res, 'unexpected_keys', None) or (load_res.get('unexpected_keys') if isinstance(load_res, dict) else None)
        if args.verbose:
            print("load_state_dict() result:", load_res)
        if missing:
            print("Missing keys (not found in checkpoint):")
            for k in missing[:30]:
                print("  ", k)
            if len(missing) > 30: print("  ... and", len(missing)-30, "more")
        if unexpected:
            print("Unexpected keys (in checkpoint but not in model):")
            for k in unexpected[:30]:
                print("  ", k)
            if len(unexpected) > 30: print("  ... and", len(unexpected)-30, "more")
    except Exception as e:
        raise RuntimeError(f"Failed to load state_dict into reconstructed model: {e}")

    print("Model loaded successfully (or partially with strict=False).")

    # Check for target stats in extras
    target_mean = None
    target_std = None
    if not args.no_denorm and isinstance(extras, dict):
        if 'target_mean' in extras:
            target_mean = extras['target_mean']
        if 'target_std' in extras:
            target_std = extras['target_std']
    # normalize to numpy arrays if tensors
    def _to_np(x):
        if x is None: return None
        if isinstance(x, torch.Tensor): return x.cpu().numpy()
        return np.array(x)
    target_mean = _to_np(target_mean)
    target_std = _to_np(target_std)

    # Predict each image
    for img_path in args.images:
        if not os.path.isfile(img_path):
            print("File not found:", img_path); continue
        raw_pred = predict_image(model, img_path, device)
        if target_mean is not None and target_std is not None:
            den = denormalize(raw_pred, target_mean, target_std)
            print(f"Image: {img_path}")
            print("  Pred (normalized):", raw_pred)
            # labeled output
            print(f"  Pred (denormalized): predicted_fuel_flow = {den[0]:.6f}, predicted_diluent_flow = {den[1]:.6f}")
        else:
            print(f"Image: {img_path}")
            print("  Pred (raw):", raw_pred)
            if not args.no_denorm:
                print("  (No target_mean/target_std found in checkpoint; provide them if needed.)")
        print("-" * 60)

    print("Inference done.")

if __name__ == '__main__':
    main()
