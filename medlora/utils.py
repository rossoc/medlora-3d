# medlora/utils.py
import os, json, random, time, yaml
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p: Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _make_serializable(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (list, tuple)):
        return [ _make_serializable(v) for v in x ]
    if isinstance(x, dict):
        return { k: _make_serializable(v) for k, v in x.items() }
    return x

def save_yaml(d: dict, path: Path):
    d = _make_serializable(d)
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)

def save_json(d: dict, path: Path):
    d = _make_serializable(d)
    with open(path, "w") as f: json.dump(d, f, indent=2)

def plot_curves(train_losses, val_losses, val_dices, outdir: Path):
    ensure_dir(outdir)
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.grid(True); plt.tight_layout()
    plt.savefig(Path(outdir)/"loss_curve.png", dpi=120); plt.close()

    plt.figure()
    plt.plot(val_dices, label="val_dice")
    plt.xlabel("epoch"); plt.ylabel("dice"); plt.legend()
    plt.grid(True); plt.tight_layout()
    plt.savefig(Path(outdir)/"dice_curve.png", dpi=120); plt.close()
