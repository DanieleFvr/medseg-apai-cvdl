import torch
from tqdm import tqdm

from src.ssl import make_ssl_batch


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
    if not 0.0 <= ssl_loss_weight <= 1.0:
        raise ValueError("ssl_loss_weight should be a value between 0.0 and 1.0 included.")
    use_ssl = ssl_loss_weight > 0.0

    model.train()
    running_loss = 0.0
    count_batches = 0

    # For each batch:
    for imgs, masks in tqdm(loader, desc="Training", leave=False):
        imgs, masks = imgs.to(device), masks.to(device).float()
        optimizer.zero_grad()
        logits = model(imgs, is_seg=True)  # Forward pass (segmentation)
        loss = criterion(logits, masks)

        if use_ssl:
            imgs_ssl, labels_ssl = make_ssl_batch(imgs, center_crop)
            imgs_ssl = imgs_ssl.to(device)
            labels_ssl = labels_ssl.to(device)
            logits_ssl = model(imgs_ssl, is_seg=False)  # Forward pass (SSL)
            loss_SSL = criterion_ssl(logits_ssl, labels_ssl)
            loss = loss + (loss_SSL * ssl_loss_weight)

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
        threshold,
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
    for imgs, masks in tqdm(loader, desc="Validating", leave=False):
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
        "val_fphw": val_fphw,
    }


def train_student_one_epoch(
        model_s,
        model_t,
        loader,
        criterion,
        optimizer,
        device,
):
    model_s.train()
    model_t.eval()

    running_hard_loss = 0.0
    running_soft_loss = 0.0
    running_total_loss = 0.0
    count_batches = 0

    # For each batch...
    for imgs, masks in tqdm(loader, desc="Training", leave=False):
        imgs, masks = imgs.to(device), masks.to(device).float()
        optimizer.zero_grad()

        # Teacher forward pass
        with torch.no_grad():
            logits_t = model_t(imgs, is_seg=True)

        # Student forward pass
        logits_s = model_s(imgs)

        loss, parts = criterion(logits_s, logits_t, masks)  # Compute student loss
        loss.backward()
        optimizer.step()

        # Bookkeeping
        running_hard_loss += parts["hard_loss"].item()
        running_soft_loss += parts["soft_loss"].item()
        running_total_loss += loss.item()
        count_batches += 1

    # Return dict with avg train hard and soft losses in an epoch for logging
    return {
        "train_hard_loss": running_hard_loss / max(count_batches, 1),
        "train_soft_loss": running_soft_loss / max(count_batches, 1),
        "train_total_loss": running_total_loss / max(count_batches, 1)
    }


@torch.no_grad()
def validate_student_one_epoch(
        model,
        loader,
        criterion,
        device,
        threshold,
        eps=1e-6,
):
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
    for imgs, masks in tqdm(loader, desc="Validating", leave=False):
        # Moving tensors to device
        imgs, masks = imgs.to(device), masks.to(device).float()

        # Forward pass
        logits = model(imgs)

        # Compute loss
        loss = criterion(logits, masks)
        running_loss += loss.item()
        count_batches += 1

        probs = torch.sigmoid(logits)  # Convert logits to binary predictions (B, 1, H, W)
        preds = (probs > threshold).float()  # Apply threshold (B, 1, H, W)

        # Identify pos/neg samples by gt
        is_pos = (masks.sum(dim=(1, 2, 3)) > 0)  # Bool tensor (B,)
        is_neg = ~is_pos  # Bool tensor (B,)

        # Only compute DSC and IoU if there is at least one positive image in the batch
        if is_pos.any():
            # Selecting positive samples
            p = preds[is_pos]  # (B_pos, 1, H, W)
            g = masks[is_pos]  # (B_pos, 1, H, W)

            # Flattening
            p = p.flatten(start_dim=1)  # (B_pos, H*W)
            g = g.flatten(start_dim=1)  # (B_pos, H*W)

            intersection = (p * g).sum(dim=1)  # Compute intersection per sample (B_pos,), TP pixels

            # Compute sums per sample
            p_sum = p.sum(dim=1)  # (B_pos,), predicted foreground pixels
            g_sum = g.sum(dim=1)  # (B_pos,), GT foreground pixels

            union = (p + g - p * g).sum(dim=1)  # compute union per sample (B_pos), TP + FP + FN
            dsc = (2.0 * intersection + eps) / (p_sum + g_sum + eps)  # Compute DSC (B_pos,)
            iou = (intersection + eps) / (union + eps)  # Compute IoU (B_pos)

            # Aggregate
            dsc_sum += dsc.sum().item()
            iou_sum += iou.sum().item()
            pos_count += dsc.numel()

        # Compute FPIR and FP/HW
        if is_neg.any():
            preds_neg = preds[is_neg]

            # ----------------- FPIR ON NEGATIVES ONLY

            neg_total += is_neg.sum().item()  # Add number of negatives samples in the batch to bookkeping
            neg_fp += (preds_neg.sum(dim=(1, 2, 3)) > 0).sum().item()  # Add number of false positives to bookkeeping

            # ----------------- FP/HW ON NEGATIVS ONLY

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
        "val_fphw": val_fphw,
    }
