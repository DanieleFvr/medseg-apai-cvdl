import torch
from tqdm import tqdm

from src.ssl_rotation_prediction import make_ssl_batch


def train_teacher_one_epoch(
        model,
        loader,
        criterion,
        criterion_ssl,
        ssl_loss_weight,
        optimizer,
        device,
        center_crop,
):
    """
    This function trains the teacher model for one epoch.

    Returns:
        Loss averaged over all batches in one epoch.
    """
    model.train()
    running_loss = 0.0
    count_batches = 0

    # For each batch:
    for imgs, masks in tqdm(loader, desc="train", leave=False):  # TODO is it ok to leave tqdm?
        imgs, masks = imgs.to(device), masks.to(device).float()
        imgs_ssl, labels_ssl = make_ssl_batch(imgs, center_crop)
        imgs_ssl = imgs_ssl.to(device)
        labels_ssl = labels_ssl.to(device)
        optimizer.zero_grad()
        logits = model(imgs, is_seg=True)  # Forward pass (segmentation)
        logits_ssl = model(imgs_ssl, is_seg=False)  # Forward pass (SSL)

        # Compute loss
        loss_SEG = criterion(logits, masks)  # TODO inconsistent naming style, lowercase
        loss_SSL = criterion_ssl(logits_ssl, labels_ssl)
        loss = loss_SEG + (loss_SSL * ssl_loss_weight)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        count_batches += 1

    return running_loss / max(count_batches, 1)


@torch.no_grad()
def validate_teacher_one_epoch(
        model,
        loader,
        criterion,
        device,
        threshold=0.5,  # TODO why is this only 0.5? Check where it's called
        eps=1e-6,
):
    """
    This function computes one epoch of validation for the teacher model.

    Returns:
        Dictionary with loss and metrics.
    """
    model.eval()

    # Loss bookkeeping
    running_loss = 0.0
    count_batches = 0

    # Metrics bookkeeping
    dsc_sum = 0.0  # Sum of DSC over positive samples
    iou_sum = 0.0  # Sum of IoU over positive samples
    pos_count = 0  # Number of positive samples evaluated
    neg_total = 0  # Number of negative samples evaluated
    neg_fp = 0  # How many of those had false positives (thresholded)
    neg_fphw_sum = 0.0  # Sum over negative samples of FP pixels/HW (thresholded)

    # For each batch...
    for imgs, masks in tqdm(loader, desc="val", leave=False):
        imgs, masks = imgs.to(device), masks.to(device).float()

        # Forward pass
        logits_SEG = model(imgs, is_seg=True)  # TODO lowercase variable

        # Compute loss
        loss = criterion(logits_SEG, masks)
        running_loss += loss.item()
        count_batches += 1

        probs = torch.sigmoid(logits_SEG)  # Convert logits to binary predictions (B, 1, H, W)
        preds = (probs > threshold).float()  # Apply threshold (B, 1, H, W)

        # Identify pos/neg samples by gt
        is_pos = (masks.sum(dim=(1, 2, 3)) > 0)  # bool tensor (B,)
        is_neg = ~is_pos  # bool tensor (B,)

        # Only compute DSC and IoU if there is at least one positive image in the batch
        if is_pos.any():
            # Selecting positive samples
            p = preds[is_pos]  # (B_pos, 1, H, W)
            g = masks[is_pos]  # (B_pos, 1, H, W)

            # Flattening
            p = p.flatten(start_dim=1)  # (B_pos, H*W)
            g = g.flatten(start_dim=1)  # (B_pos, H*W)

            intersection = (p * g).sum(dim=1)  # Compute intersection per sample, TP pixels (B_pos,)

            # Compute sums per sample
            p_sum = p.sum(dim=1)  # Predicted foreground pixels (B_pos,)
            g_sum = g.sum(dim=1)  # GT foreground pixels (B_pos,)

            union = (p + g - p * g).sum(dim=1)  # Compute union per sample, TP + FP + FN (B_pos)
            dsc = (2.0 * intersection + eps) / (p_sum + g_sum + eps)  # Compute DSC (B_pos,)
            iou = (intersection + eps) / (union + eps)  # Compute IoU (B_pos)

            # Aggregate
            dsc_sum += dsc.sum().item()
            iou_sum += iou.sum().item()
            pos_count += dsc.numel()

        # Compute FPIR and FP/HW
        if is_neg.any():
            preds_neg = preds[is_neg]

            # Compute FPIR, on negatives only
            neg_total += is_neg.sum().item()  # Add number of negatives samples in the batch to bookkeeping
            neg_fp += (preds_neg.sum(dim=(1, 2, 3)) > 0).sum().item()  # Add number of false positives to bookkeeping

            # Compute FP/HW, on negatives only
            neg_frac = preds_neg.flatten(start_dim=1)  # Flatten (B_neg, H*W)
            neg_frac = neg_frac.mean(dim=1)  # Mean over H*W, total pixels (B_neg,)
            neg_fphw_sum += neg_frac.sum().item()  # Add to bookkeeping

    # Define metrics per epoch
    val_loss = running_loss / max(count_batches, 1)
    val_dsc = dsc_sum / max(pos_count, 1)
    val_iou = iou_sum / max(pos_count, 1)
    val_fpir = neg_fp / max(neg_total, 1)
    val_fphw = neg_fphw_sum / max(neg_total, 1)

    return {
        "val_loss": val_loss,
        "val_dsc": val_dsc,
        "val_iou": val_iou,
        "val_fpir": val_fpir,
        "val_fphw": val_fphw
    }
