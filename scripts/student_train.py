import numpy as np
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

from src import data, models, losses, loops, schedules, optimizers, utils
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
criterion_s = losses.TotalKDLoss(
    hard_loss=hard_loss,
    soft_loss=soft_loss,
    w_soft=cfg.soft_loss_weight,
).to(cfg.device)

model_s = models.StudentUNet(num_groups=cfg.student_num_groups).to(cfg.device)  # Initialize model

# Initialize student optimizer
student_optimizer = optimizers.build_adam_optimizer(
    model=model_s,
    lr=cfg.lr_student,
)

# Load trained teacher checkpoint
teacher_checkpoint_path = Path(cfg.ckpt_dir / cfg.teacher_checkpoint_file_name)

# Load teacher checkpoint
trained_teacher_ckpt = torch.load(teacher_checkpoint_path, map_location=cfg.device)
model_t = models.TeacherUNet(num_groups=cfg.num_groups).to(cfg.device)
model_t.load_state_dict(trained_teacher_ckpt["model_state_dict"], strict=True)

model_t.eval()
for p in model_t.parameters():
    p.requires_grad_(False)

# Make checkpoint directory if not already present
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ckpt_path = Path(cfg.ckpt_dir / f"{timestamp}_student_run")
ckpt_path.mkdir(parents=True, exist_ok=True)

train_hard_losses: list[float] = []
val_losses: list[float] = []
best_val_loss: float = float("inf")

for epoch in range(cfg.num_epochs_student):
    # Apply OHEM warmup schedule
    criterion_s.hard_loss.neg_ohem_weight = schedules.ohem_warmup_schedule(
        epoch=epoch,
        warmup_active=cfg.student_ohem_warmup_active,
        activation_epoch=cfg.student_ohem_activation_epoch,
        final_weight=cfg.student_ohem_final_weight,
    )
    # Run training
    train_loss = loops.train_student_one_epoch(
        model_s=model_s,
        model_t=model_t,
        loader=train_loader,
        criterion=criterion_s,
        optimizer=student_optimizer,
        device=cfg.device,
    )
    train_total_loss = train_loss["train_total_loss"]
    train_hard_loss = train_loss["train_hard_loss"]
    train_soft_loss = train_loss["train_soft_loss"]

    train_hard_losses.append(train_loss["train_hard_loss"])

    for t in cfg.thresholds:
        # Run validation
        val_results = loops.validate_student_one_epoch(
            model=model_s,
            loader=val_loader,
            criterion=hard_loss,
            device=cfg.device,
            threshold=t,  # TODO fix that threshold = 0.5 issue, remember teacher_train.py doesn't pass this parameter
        )

        if t == cfg.thresholds[0]:
            # Get the val loss for that epoch
            # Note: any threshold is fine, but only one must be chosen; I chose the first
            val_loss = val_results["val_loss"]
            val_losses.append(val_loss)  # Append loss

            utils.print_epoch_info(
                val_summary=f"VAL LOSS={val_results['val_loss']:.4f}",
                train_summary=(
                    f"HARD={train_loss['train_hard_loss']:.4f}, "
                    f"SOFT={train_loss['train_soft_loss']:.4f}, "
                    f"TOT={train_loss['train_total_loss']:.4f}"
                ),
                epoch=epoch,
                num_epochs=cfg.num_epochs_student,
                optimizer=student_optimizer,
                neg_ohem_weight=criterion_s.hard_loss.neg_ohem_weight,  # hard_loss specifically, because it needs to
                # pass the current OHEM weight
            )
        utils.print_epoch_metrics(
            threshold=t,
            val_results=val_results,
        )

    # Save chekpoints
    best_val_loss = utils.save_checkpoint(
        ckpt_path=ckpt_path,
        epoch=epoch,
        model=model_s,
        optimizer=student_optimizer,
        best_val_loss=best_val_loss,
        val_results=val_results,
    )