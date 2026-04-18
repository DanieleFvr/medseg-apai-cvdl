from pathlib import Path

# Paths
manifest_path = Path(f"/content/drive/MyDrive/APAI_CVDL_shared/project/meta/manifest.parquet")
LOCAL_ROOT = Path("/content/project_data/data/pre/img/")
ROOT = Path("/content/")  # TODO rename, not descriptive enough
LOCAL_MANIFEST = Path(f"{ROOT}/manifest_local.parquet")


# Dataloaders
oversample: bool = True  # Oversampling flag
k: float  = 0.5  # Oversampling multiplier
