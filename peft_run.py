import time
import argparse
from monai.networks.nets.swin_unetr import SwinUNETR
from peft import LoraConfig, get_peft_model
from load import load_ct_ssl_encoder
from train import train
from data import build_loaders, get_channels
from typing import Sequence
import torch

MSD_TASKS: Sequence[str] = [
    "Task01_BrainTumour",
    "Task02_Heart",
    "Task03_Liver",
    "Task04_Hippocampus",
    "Task05_Prostate",
    "Task06_Lung",
    "Task07_Pancreas",
    "Task08_HepaticVessel",
    "Task09_Spleen",
    "Task10_Colon",
]


def run(
    save_dir,
    dataset,
    train_fraction,
    seed,
    batch_size,
    num_workers,
    epochs,
    lr,
    wd,
    early_stopping,
    patience,
    min_epochs,
):
    train_loader, val_loader, train_eval_loader, _ = build_loaders(
        save_dir,
        dataset,
        train_fraction,
        seed,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    in_ch, out_ch = get_channels(save_dir, dataset)

    model = SwinUNETR(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=out_ch,
        feature_size=48,
        use_checkpoint=True,
        use_v2=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["qkv", "proj", "linear1", "linear2"],
        bias="none",
        modules_to_save=None,
    )

    _, _ = load_ct_ssl_encoder(model)

    peft_model = get_peft_model(model, lora_config)  # type: ignore

    peft_model.print_trainable_parameters()

    start = time.time()
    logs, best_state = train(
        model=model,
        loaders=(train_loader, val_loader, train_eval_loader),
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_epochs=epochs,
        lr=lr,
        wd=wd,
        early_stopping=early_stopping,
        patience=patience,
        min_epochs=min_epochs,
    )
    wall = time.time() - start
    return wall, logs, best_state


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", default="content/data")
    p.add_argument("--dataset", choices=MSD_TASKS, required=True)
    p.add_argument("--train-fraction", type=int, choices=[5, 20, 80, 100], default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--early-stopping", default="True")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-epochs", type=int, default=3)
    # p.add_argument("--model", choices=["swinv1", "swinv2"], required=True)
    # p.add_argument("--method", choices=["fft", "lora"], required=True)
    # p.add_argument("--splits-dir", type=Path, default=Path("splits"))
    # p.add_argument("--lora-r", type=int, default=8)
    # p.add_argument("--lora-alpha", type=int, default=16)
    # p.add_argument("--lora-dropout", type=float, default=0.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(**vars(args))
