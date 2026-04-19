from pathlib import Path

import torchvision.transforms as T

# Paths
manifest_path: str | Path = Path(f"/content/drive/MyDrive/APAI_CVDL_shared/project/meta/manifest.parquet")
LOCAL_ROOT: str | Path = Path("/content/project_data/data/pre/img/")  # TODO should be lowercase
ROOT: str | Path = Path("/content/")  # TODO rename, not descriptive enough, and should be lowercase
LOCAL_MANIFEST: str | Path = Path(f"{ROOT}/manifest_local.parquet")  # TODO should be lowercase

# Dataloaders
oversample: bool = True  # Oversampling flag
k: float  = 0.5  # Oversampling multiplier

# Rotation prediction SSL
center_crop = T.CenterCrop(size=384) # TODO find data type

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Active learning sample selection
K: int = 800
batch_size: int = 8

# Teacher model
num_groups: int = 8  # Number of groups for GroupNorm