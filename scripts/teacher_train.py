from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

import dataframe
import dataset
import config.config as cfg
import src.teacher_train
import model
from loss import FocalBCEDiceLoss
from dataset import PneumoDatasetForAL
from src.teacher_train import ohem_warmup_schedule
import src.teacher_train
import src.active_learning


# Load the project manifest into a DataFrame.
df = pd.read_parquet(cfg.manifest_path)

# Create local copy of the manifest, used for the active learning loop.
LOCAL_MANIFEST = df.to_parquet(cfg.LOCAL_MANIFEST, index=False)  # TODO rename local manifest variable

# Create split training and validation DataFrames
df_train, df_val = dataframe.dataframe_split(df)  # TODO suboptimal naming
train_ds, val_ds = dataset.build_datasets(df_train, df_val, cfg.ROOT)  # Create split training and validation Dataset instances

# Build training and validation dataloaders, given config parameters
train_loader, val_loader = dataset.build_dataloaders(
    train_ds,
    val_ds,
    df_train,
    oversample=cfg.oversample,
    k=cfg.k,
)

center_crop = T.CenterCrop(size=cfg.center_crop_size)

# Initialize teacher model criterion instance
criterion = FocalBCEDiceLoss(
    cfg.dice_weight,
    cfg.alpha,
    cfg.gamma,
    cfg.neg_ohem_weight,
    cfg.neg_topk,
).to(cfg.device)
criterion_ssl = nn.CrossEntropyLoss().to(cfg.device)  # Initialize SSL criterion instance

model = model.UNet(num_classes=1).to(cfg.device)  # Initialize model

# Initialize optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.lr,
)

train_losses: list[float] = []
val_losses: list[float] = []

# Calculating indexes for the very first round of AL
ROUND_ID: int = 1
df_new, selected_image_ids = src.active_learning.select_next_round_uncertainty(
    model=model,
    local_manifest=cfg.LOCAL_MANIFEST,
    round_id=ROUND_ID,
    K=cfg.K,
    batch_size=8,
)

df_new.to_parquet(LOCAL_MANIFEST, index=False)  # Write the manifest to select them  #TODO fix comment

for r in range(cfg.ROUNDS):
    # Build training and validation Datasets
    train_L_ds = src.dataset.PneumoDatasetForAL(
        LOCAL_MANIFEST,
        split="train",
        status="L",
        project_root=Path("project_data"),  # TODO this should be in config
    )
    val_ds = src.dataset.PneumoDatasetForAL(
        LOCAL_MANIFEST,
        split="val",
        project_root=Path("project_data"),
    )
    # Build training and validation DataLoaders
    train_loader = src.dataset.DataLoader(train_L_ds, batch_size=8, shuffle=True)
    val_loader = src.dataset.DataLoader(val_ds, batch_size=8, shuffle=False)

    for epoch in range(cfg.num_epochs):
        # Apply OHEM warmup schedule
        criterion.neg_ohem_weight = src.teacher_train.ohem_warmup_schedule(
            round_id=r,
            epoch=epoch,
            activation_epoch=cfg.ohem_activation_epoch,
        )
        # Run training loop
        train_loss = src.teacher_train.train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            criterion_ssl=criterion_ssl,
            ssl_loss_weight=cfg.ssl_loss_weight,
            optimizer=optimizer,
            device=cfg.device,
        )
        train_losses.append(train_loss)

        # Validation is run for every one of the thresholds that have been chosen in config, and all results are logged
        for t in thresholds:
            # Run validation
            val_results = src.teacher_train.validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=cfg.device,
                threshold=t,
            )
            # Print metrics to console every epoch
            src.teacher_train.print_metrics(
                threshold=t,
                thresholds=thresholds,
                val_results=val_results,
                val_losses=val_losses,
                train_loss=train_loss,
                epoch=epoch,
                num_epochs=cfg.num_epochs,
                optimizer=optimizer,
                criterion=criterion,
            )
    ROUND_ID += 1
    K = cfg.k_after_first_round

    df_new, selected_image_ids = src.active_learning.select_next_round_uncertainty(model, cfg.LOCAL_MANIFEST, ROUND_ID, K=K, batch_size=8)
    df_new.to_parquet(cfg.LOCAL_MANIFEST, index=False)
