from pathlib import Path

import torch


# Paths  # TODO make all strings
manifest_path: str | Path = Path(f"/content/drive/MyDrive/APAI_CVDL_shared/project/meta/manifest.parquet")
LOCAL_ROOT: str | Path = Path("/content/data/pre/img/")  # TODO should be lowercase
ROOT: str | Path = Path("/content/")  # TODO rename, not descriptive enough, and should be lowercase
LOCAL_MANIFEST: str | Path = Path(f"{ROOT}/manifest_local.parquet")  # TODO should be lowercase
ckpt_dir: Path = Path("/content/drive/MyDrive/APAI_CVDL_shared/checkpoints/")

# DataLoaders
oversample: bool = True  # Oversampling flag
k: float  = 0.5  # Oversampling multiplier

# Rotation prediction SSL
center_crop_size: int = 384

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Active learning sample selection
batch_size: int = 8

# Teacher model
num_groups: int = 8  # Number of groups for GroupNorm

# Teacher model criterion parameters
dice_weight: float = 1.0
alpha: float = 0.25
gamma: float = 2.0
neg_ohem_weight: float = 0.05
neg_topk: int = 1024

# Teacher training hyperparameters
lr_teacher: float = 2.5e-5  # Starting LR
ssl_loss_weight: float = 0.1
num_epochs: int = 20
ROUNDS: int = 5  # Number of AL rounds
K: int = 3000  # Number of samples for the first AL round
k_after_first_round: int = 200  # Number of samples selected on every round after the first
ohem_activation_epoch: int = 4

# Validation config values
thresholds: list[float] = [0.25, 0.5, 0.7]

# Student training hyperparameters
lr_student: float = 5e-5  # Starting LR