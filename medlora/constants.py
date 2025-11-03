from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Sequence

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

CT_TASKS = frozenset(
    [
        "Task03_Liver",
        "Task06_Lung",
        "Task07_Pancreas",
        "Task08_HepaticVessel",
        "Task09_Spleen",
        "Task10_Colon",
    ]
)
MRI_TASKS = frozenset(
    [
        "Task01_BrainTumour",
        "Task02_Heart",
        "Task04_Hippocampus",
        "Task05_Prostate",
    ]
)

DEFAULT_VAL_FRAC = 0.2
DEFAULT_ROI: Tuple[int, int, int] = (96, 96, 96)

SSL_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/"
    "ssl_pretrained_weights.pth"
)

SSL_PATH = os.environ.get(
    "MEDLORA_SSL_PATH", str(Path.home() / ".medlora" / "ssl_pretrained_weights.pth")
)
