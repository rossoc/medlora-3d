from monai.networks.nets.swin_unetr import SwinUNETR
from pathlib import Path
from monai.apps.utils import download_url
import os
import torch
import re


SSL_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/"
    "ssl_pretrained_weights.pth"
)

SSL_PATH = os.environ.get(
    "MEDLORA_SSL_PATH", str(Path.home() / ".medlora" / "ssl_pretrained_weights.pth")
)


def ensure_ssl(path=SSL_PATH, url=SSL_URL):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(path).exists():
        download_url(url, path)
    return path


def load_ct_ssl_encoder(model: SwinUNETR, path=SSL_PATH):
    """
    Load public CT-SSL encoder weights into Swin-UNETR.
    Handles older key layouts: prefixes and layersX.a.b -> layersX.a.
    Returns (loaded_tensors_count, total_encoder_tensors).
    """
    ensure_ssl(path)
    blob = torch.load(path, map_location="cpu")
    for k in ("state_dict", "model", "weights", "params", "net"):
        if isinstance(blob, dict) and k in blob and isinstance(blob[k], dict):
            blob = blob[k]
            break

    enc_sd = model.swinViT.state_dict()  # relative keys

    def rel_map(k: str) -> str:
        for p in (
            "module.",
            "backbone.",
            "encoder.",
            "swin_vit.",
            "swinViT.",
            "network.",
        ):
            if k.startswith(p):
                k = k[len(p) :]
        # collapse layersX.a.b.* -> layersX.a.*
        k = re.sub(r"^(layers[1-4])\.([0-9]+)\.([0-9]+)\.", r"\1.\2.", k)
        return k

    to_load = {}
    for ck, cv in blob.items() if isinstance(blob, dict) else []:
        if not isinstance(cv, torch.Tensor):
            continue
        rel = rel_map(ck)
        if rel in enc_sd and enc_sd[rel].shape == cv.shape:
            to_load[rel] = cv

    if to_load:
        enc_new = enc_sd.copy()
        enc_new.update(to_load)
        model.swinViT.load_state_dict(enc_new, strict=False)

    return len(to_load), len(enc_sd)
