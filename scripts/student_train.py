import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src import data, models, losses
from config import config as cfg


# Load original manifest
df = pd.read_parquet(cfg.manifest_path)

# Create split training and validation DataFrames based on the subset labeled during active learning
df_labeled = df[df["round_added"] != -1].copy()
df_train_s, df_val_s = train_test_split(
    df_labeled,
    test_size=0.2,
    random_state=7,
    stratify=df_labeled["class"]  # Preserve natural distribution of the df_labeled for both train and val splits
)

# Create split training and validation Dataset instances
train_ds, val_ds = data.build_datasets(
    df_train=df_train_s,
    df_val=df_val_s,
    ROOT=cfg.ROOT,
)

# Build training and validation dataloaders
train_loader, val_loader = data.build_dataloaders(
    train_ds=train_ds,
    val_ds=val_ds,
    df_train=df_labeled,
    oversample=False,
)  # TODO what's the best practice for the missing k param?

model = models.StudentUNet(num_groups=cfg.student_num_groups).to(cfg.device)  # Initialize model

# Compute hard loss
hard_loss = losses.FocalBCEDiceLoss(
    dice_weight=cfg.hard_loss_dice_weight,
    alpha=cfg.hard_loss_alpha,
    gamma=cfg.hard_loss_gamma,
    neg_ohem_weight=cfg.hard_loss_neg_ohem_weight,
    neg_topk=cfg.hard_loss_neg_topk,
).to(cfg.device)

# Compute soft loss
soft_loss = losses.SoftKDLoss(T=cfg.soft_loss_temperature).to(cfg.device)

# Initialize student model criterion instance
criterion = losses.TotalKDLoss(
    hard_loss=hard_loss,
    soft_loss=soft_loss,
    w_soft=cfg.soft_loss_weight,
).to(cfg.device)