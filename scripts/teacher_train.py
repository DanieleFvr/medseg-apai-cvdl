import pandas as pd

import torch
import torch.nn as nn

import dataframe
import dataset
import config.config as cfg
from model import UNet
from loss import FocalBCEDiceLoss


# Load the project manifest into a DataFrame.
df = pd.read_parquet(cfg.manifest_path)

# Create local copy of the manifest, used for the active learning loop.
LOCAL_MANIFEST = df.to_parquet(cfg.LOCAL_MANIFEST, index=False)  # TODO rename local manifest

# Create split training and validation DataFrames
df_train, df_val = dataframe.dataframe_split(df)  # TODO suboptimal naming
train_ds, val_ds = dataset.build_datasets(df_train, df_val, cfg.ROOT)  # Create split training and validation Dataset instances

# Build training and validation dataloaders, given config parameters
train_loader, val_loader = dataset.build_dataloaders(
    train_ds,
    val_ds,
    df_train,
    oversample=cfg.oversample,
    k=cfg.k
)

center_crop = T.CenterCrop(size=cfg.center_crop_size)

# Initialize teacher model criterion instance
criterion = FocalBCEDiceLoss(
    cfg.dice_weight,
    cfg.alpha,
    cfg.gamma,
    cfg.neg_ohem_weight,
    cfg.neg_topk
).to(cfg.device)
criterion_ssl = nn.CrossEntropyLoss().to(cfg.device)  # Initialize ssl head criterion instance

model = UNet(num_classes=1).to(cfg.device)  # Initialize model
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)  # Initialize optimizer

