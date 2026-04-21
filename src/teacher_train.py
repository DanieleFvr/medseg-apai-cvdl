import torch
from tqdm import tqdm

import config.config as cfg
from ssl_rotation_prediction import make_ssl_batch


def train_one_epoch(model, loader, criterion, criterion_ssl, ssl_loss_weight, optimizer, device):
    """
    This function trains the model for one epoch.

    Returns:
        Loss averaged over all batches in one epoch.
    """
    model.train()
    running_loss = 0.0
    count_batches = 0

    # For each batch:
    for imgs, masks in tqdm(loader, desc="train", leave=False):  # TODO is it ok to leave tqdm?
        imgs, masks = imgs.to(device), masks.to(device).float()
        imgs_ssl, labels_ssl = make_ssl_batch(imgs)
        imgs_ssl=imgs_ssl.to(device)
        labels_ssl=labels_ssl.to(device)
        optimizer.zero_grad()
        logits = model(imgs, is_seg=True)  # Forward pass (segmentation)
        logits_ssl = model(imgs_ssl, is_seg=False)  # Forward pass (ssl)

        # Compute loss
        loss_SEG = criterion(logits, masks)  # TODO inconsistent naming style, lowercase
        loss_SSL = criterion_ssl(logits_ssl, labels_ssl)
        loss = loss_SEG + (loss_SSL * ssl_loss_weight)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        count_batches += 1

    return running_loss / max(count_batches, 1)