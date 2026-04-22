from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import PneumoDatasetForAL


def entropy_score_from_logits(logits: torch.Tensor) -> torch.Tensor:  # TODO fix docstring, too generic
    """
    This function is used to calculate the entropy score given logits.

    Args:
        logits (torch.Tensor): logits tensor
    Returns:

    """
    probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    ent = -(probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs))
    return ent.mean(dim=(1, 2, 3))


@torch.no_grad()
def select_next_round_uncertainty(
        model,
        local_manifest,
        round_id: int,
        device,
        K: int,
        batch_size: int,
) -> tuple[pd.DataFrame, list] | None:
    """  # TODO write docstring

    Args:
        model (torch.nn.Module): the model used for teacher training.
        local_manifest (PneumoDatasetForAL): the local dataset used for teacher training.  # TODO sure? And fix type
        round_id (int): the round id of the current active learning round, used only for manifest logging.
        device (torch.device): the device used for training.
        K (int) = how many samples are to be selected each.
        batch_size (int): the batch size used for training each round.
    Returns:
        tuple[pd.DataFrame, list] | None:
            df (pd.DataFrame): the updated local dataframe.
            selected_image_ids (list): the IDs of the images selected for the current active learning round.
            The function returns None if there are no unlabeled samples left in training pool.
    """
    pos_to_neg = (1, 2)
    df = pd.read_parquet(local_manifest)  # TODO are we calling the remote and local manifests with the same var name?
    pool = df[(df["split"] == "train") & (df["status"].astype(str) == "U")].copy()

    if len(pool) == 0:
        print("No unlabeled samples left in train pool")
        return None

    pool_indices = pool.index.to_numpy()
    ds_u = PneumoDatasetForAL(local_manifest, pool_indices, project_root=Path("/content/project_data"))
    loader_u = DataLoader(ds_u, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()
    all_idx = []
    all_score = []
    pbar = tqdm(loader_u, desc=f" [calculating indexes]")

    for idx_batch, x in pbar:
        x = x.to(device)
        logits = model(x, is_seg=True)
        ent = entropy_score_from_logits(logits)
        all_idx.extend(idx_batch.cpu().numpy().tolist())
        all_score.extend(ent.detach().cpu().numpy().tolist())

    scores = np.array(all_score)
    idxs = np.array(all_idx)
    a, b = pos_to_neg
    K_eff = min(K, len(scores))
    y = df.loc[idxs, "class"].to_numpy(dtype=int)  # Read the real class from the manifest
    idxs_pos = idxs[y == 1]  # Split in 2 groups, positive and negative indexes
    sc_pos = scores[y == 1]
    idxs_neg = idxs[y == 0]
    sc_neg = scores[y == 0]
    K_pos = (K_eff * a) // (a + b)
    K_neg = K_eff - K_pos

    # Handle class scarcity: cap and refill with the other class
    K_pos = min(K_pos, len(idxs_pos))
    K_neg = min(K_neg, len(idxs_neg))
    rem = K_eff - (K_pos + K_neg)

    # Fill remaining from whichever class still has items
    if rem > 0:
        if len(idxs_pos) - K_pos >= len(idxs_neg) - K_neg:
            K_pos = min(len(idxs_pos), K_pos + rem)
        else:
            K_neg = min(len(idxs_neg), K_neg + rem)

    # Pick top uncertain within each class
    if K_pos > 0:
        top_pos = np.argpartition(-sc_pos, K_pos - 1)[:K_pos]
        sel_pos = idxs_pos[top_pos]
    else:
        sel_pos = np.array([], dtype=idxs.dtype)

    if K_neg > 0:
        top_neg = np.argpartition(-sc_neg, K_neg - 1)[:K_neg]
        sel_neg = idxs_neg[top_neg]
    else:
        sel_neg = np.array([], dtype=idxs.dtype)

    selected_manifest_indices = np.concatenate([sel_pos, sel_neg])
    np.random.shuffle(selected_manifest_indices)  # Optional shuffle to avoid ordering by class
    df.loc[selected_manifest_indices, "status"] = "L"
    df.loc[selected_manifest_indices, "round_added"] = int(round_id)
    selected_image_ids = df.loc[selected_manifest_indices, "image_id"].to_list()
    return df, selected_image_ids
