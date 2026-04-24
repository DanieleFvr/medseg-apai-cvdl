from pathlib import Path

import pandas as pd
import torch
from torchvision import transforms as T
import torch.nn as nn
from torch.utils.data import DataLoader

from src import data, active_learning, loss, model, teacher_train
from config import config as cfg


# Load the project manifest into a DataFrame.
df = pd.read_parquet(cfg.manifest_path)  # TODO unused?

# Create local copy of the manifest, used for the active learning loop.
df.to_parquet(cfg.LOCAL_MANIFEST, index=False)  # TODO unused?

# Create split training and validation DataFrames
df_train, df_val = data.dataframe_split(df=df)  # TODO unused

# Create split training and validation Dataset instances
train_ds, val_ds = data.build_datasets(
    df_train=df_train,
    df_val=df_val,
    ROOT=cfg.ROOT,
)  # TODO unused, test and fix

# Build training and validation dataloaders, given config parameters
train_loader, val_loader = data.build_dataloaders(
    train_ds=train_ds,
    val_ds=val_ds,
    df_train=df_train,
    oversample=cfg.oversample,
    k=cfg.k,
)  # TODO unused

center_crop = T.CenterCrop(size=cfg.center_crop_size)

# Initialize teacher model criterion instance
criterion = loss.FocalBCEDiceLoss(
    dice_weight=cfg.dice_weight,
    alpha=cfg.alpha,
    gamma=cfg.gamma,
    neg_ohem_weight=cfg.neg_ohem_weight,
    neg_topk=cfg.neg_topk,
).to(cfg.device)
criterion_ssl = nn.CrossEntropyLoss().to(cfg.device)  # Initialize SSL criterion instance

model = model.UNet(num_groups=cfg.num_groups).to(cfg.device)  # Initialize model

# Initialize optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.lr,
)

train_losses: list[float] = []
val_losses: list[float] = []

# Calculate indexes for the very first round of AL
ROUND_ID = 1  # First round
# Rewrite the local manifest to account for the first round's sample selections
df_new, selected_image_ids = active_learning.select_next_round_uncertainty(
    model=model,
    local_manifest=cfg.LOCAL_MANIFEST,
    round_id=ROUND_ID,
    device=cfg.device,
    K=cfg.K,
    batch_size=8,
)  # TODO shouldn't this be at the beginning of the for loop? Test and maybe move it

df_new.to_parquet(cfg.LOCAL_MANIFEST, index=False)  # Write the manifest to select them  #TODO fix comment

for r in range(cfg.ROUNDS):  # TODO rename "r" to "round" for clarity
    # Build training and validation Datasets
    train_L_ds = data.PneumoDatasetForAL(
        cfg.LOCAL_MANIFEST,
        split="train",
        status="L",
        project_root=Path("/content"),  # TODO this should be in config
    )
    val_ds = data.PneumoDatasetForAL(
        cfg.LOCAL_MANIFEST,
        split="val",
        project_root=Path("/content"),
    )
    # Build training and validation DataLoaders
    train_loader = DataLoader(train_L_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    for epoch in range(cfg.num_epochs):
        # Apply OHEM warmup schedule
        criterion.neg_ohem_weight = teacher_train.ohem_warmup_schedule(
            round_id=r,
            epoch=epoch,
            activation_epoch=cfg.ohem_activation_epoch,
        )
        # Run training loop
        train_loss = teacher_train.train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            criterion_ssl=criterion_ssl,
            ssl_loss_weight=cfg.ssl_loss_weight,
            optimizer=optimizer,
            device=cfg.device,
            center_crop=center_crop,
        )
        train_losses.append(train_loss)

        # Validation is run for every one of the thresholds that have been chosen in config, and all results are logged
        for t in cfg.thresholds:
            # Run validation
            val_results = teacher_train.validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=cfg.device,
                threshold=t,
            )
            # Print metrics to console every epoch
            teacher_train.print_metrics(
                threshold=t,
                thresholds=cfg.thresholds,
                val_results=val_results,
                val_losses=val_losses,
                train_loss=train_loss,
                epoch=epoch,
                num_epochs=cfg.num_epochs,
                optimizer=optimizer,
                criterion=criterion,
            )

    ROUND_ID += 1
    df_new, selected_image_ids = active_learning.select_next_round_uncertainty(
        # TODO is selected_image_ids never used?
        model=model,
        local_manifest=cfg.LOCAL_MANIFEST,
        round_id=ROUND_ID,
        device=cfg.device,
        K=cfg.k_after_first_round,
        batch_size=8,
    )  # TODO shouldn't this too be at the beginning of the for loop? (only one)
    df_new.to_parquet(cfg.LOCAL_MANIFEST, index=False)
