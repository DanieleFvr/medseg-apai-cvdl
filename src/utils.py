from pathlib import Path

import torch


def print_epoch_info(
        val_results: dict,
        train_loss: float,
        epoch: int,
        num_epochs: int,
        optimizer,
        criterion,
) -> None:
    """
    This functions prints useful information and metrics for every epoch.

    Args:

    Returns:
        None, it prints only.
    """
    print(
        "\n-------\n"
        f"EPOCH {epoch + 1}/{num_epochs} | "
        f"TRAIN LOSS={train_loss:.4f} | "
        f"VAL LOSS={val_results['val_loss']:.4f} | "
        f"LR={optimizer.param_groups[0]['lr']} | "  # TODO where does this come from?
        f"OHEM WEIGHT={criterion.neg_ohem_weight}"
    )


def print_epoch_metrics(
        threshold: float,
        val_results: dict,
) -> None:
    """
    This functions prints useful information and metrics for every epoch.

    Args:

    Returns:
        None, it prints only.
    """
    print(
        f"THRESHOLD={threshold}\n"
        f"DSC={val_results['val_dsc']:.4f} | "
        f"IoU={val_results['val_iou']:.4f} | "
        f"FPIR={val_results['val_fpir']:.4f} | "
        f"FPHW={val_results['val_fphw']:.6f}"
    )

def save_checkpoint(
        ckpt_path: Path,
        epoch: int,
        model,
        optimizer,
        best_val_loss: float,
        val_results: dict,
):
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_results,
    }

    torch.save(checkpoint, ckpt_path / f"latest_checkpoint.pt")  # Save last checkpoint

    # Save best checkpoint if conditions are met
    if val_results["val_loss"] < best_val_loss:
        torch.save(checkpoint, ckpt_path / f"best_checkpoint_epoch_{epoch + 1:03d}.pt")
        return val_results["val_loss"]
    else:
        return best_val_loss
