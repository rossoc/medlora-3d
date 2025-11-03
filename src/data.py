from pathlib import Path
import numpy as np
import torch
from monai.apps.datasets import DecathlonDataset
from monai.data.dataset import Dataset
from monai.data.dataloader import DataLoader
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped
from monai.transforms.spatial.dictionary import (
    Orientationd,
    Spacingd,
    RandFlipd,
    RandRotate90d,
)
from monai.transforms.intensity.dictionary import (
    ScaleIntensityRanged,
    NormalizeIntensityd,
)
from monai.transforms.croppad.dictionary import (
    RandCropByPosNegLabeld,
    SpatialPadd,
)


def is_ct(task: str) -> bool:
    return task in [
        "Task03_Liver",
        "Task06_Lung",
        "Task07_Pancreas",
        "Task08_HepaticVessel",
        "Task09_Spleen",
        "Task10_Colon",
    ]


def make_transforms(task: str, roi, aug=True):
    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0) if is_ct(task) else (1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
    ]
    norm = (
        [
            ScaleIntensityRanged(
                keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
            )
        ]
        if is_ct(task)
        else [NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)]
    )

    if aug:
        extra = [
            # allow smaller crops if volume < ROI in any dim; then pad to ROI
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi,
                pos=1,
                neg=1,
                num_samples=1,
                allow_smaller=True,
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=roi),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=(0, 1, 2)),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            EnsureTyped(keys=["image", "label"]),
        ]
    else:
        # no aug for eval; keep native size (SlidingWindowInferer can handle padding internally)
        extra = [EnsureTyped(keys=["image", "label"])]

    return Compose(base + norm + extra)


def load_decathlon_list(data_dir: Path, task: str, download=True):
    data_dir = Path(data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    ds = DecathlonDataset(
        root_dir=str(data_dir),
        task=task,
        section="training",
        transform=None,  # type: ignore
        download=download,
        val_frac=0.0,
        cache_rate=0.0,
        num_workers=0,
    )
    return list(ds.data)


def train_val_split(indexes, test_size, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(indexes))
    val_count = max(1, int(test_size * len(indexes)))
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    return [indexes[i] for i in train_idx], [indexes[i] for i in val_idx]


def take_fraction(items, frac_percent, seed):
    if frac_percent >= 100:
        return items
    n = max(1, int(len(items) * (frac_percent / 100.0)))
    rng = np.random.default_rng(seed)
    keep = rng.permutation(len(items))[:n]
    return [items[i] for i in keep]


def build_loaders(
    data_dir: Path,
    task: str,
    fraction: int,
    seed: int,
    batch_size=2,
    num_workers=2,
    roi=(96, 96, 96),
    test_size=0.2,
):
    all_items = load_decathlon_list(data_dir, task, download=True)
    train_pool, val_items = train_val_split(all_items, test_size=test_size, seed=seed)
    train_items = take_fraction(train_pool, fraction, seed=seed)

    train_tf = make_transforms(task, roi, aug=True)
    eval_tf = make_transforms(task, roi, aug=False)

    train_ds = Dataset(train_items, transform=train_tf)
    val_ds = Dataset(val_items, transform=eval_tf)
    train_eval_ds = Dataset(train_items, transform=eval_tf)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return (
        train_loader,
        val_loader,
        train_eval_loader,
        {"train": train_items, "val": val_items},
    )


def get_channels(data_dir, dataset):
    props = DecathlonDataset(
        root_dir=str(data_dir),
        task=dataset,
        section="training",
        download=False,
        val_frac=0.0,
        cache_rate=0.0,
        num_workers=0,
    ).get_properties()
    in_ch = len(props.get("modality", {}) or {}) or 1
    out_ch = len(props.get("labels", {}) or {}) or 2
    return in_ch, out_ch
