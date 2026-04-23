import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd


def dataframe_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function splits a dataframe into two: one for training and one for validation, based on
    "split" col value.

    Args:
        df (pd.DataFrame): the dataframe to be split.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: the training and validation dataframes.
    """
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    return df_train, df_val


# Create dataset class
class BaselineDataset(Dataset):  # TODO refactor class name to something else: this isn't just for the baseline
    def __init__(self, df, root, img_col="img_path", mask_col="mask_path"):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.img_col = img_col
        self.mask_col = mask_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Replace slashes and join paths
        img_rel_path = row[self.img_col].replace("\\", "/")
        mask_rel_path = row[self.mask_col].replace("\\", "/")
        img_path = os.path.join(self.root, str(img_rel_path))
        mask_path = os.path.join(self.root, str(mask_rel_path))

        # Load image and mask given path AND convert them to float32
        img = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        # From (H, W) to (1, H, W)
        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask


class PneumoDatasetForAL(Dataset):
    def __init__(
            self,
            manifest_path,
            indices=None,
            status=None,
            split="train",
            project_root=None
    ):
        df_full = pd.read_parquet(manifest_path)
        self.split = split
        self.project_root = Path(project_root) if project_root is not None else None

        if indices is not None:
            # This is for the pooling dataset, where 'indices' are the original dataframe indices
            # Filter the full dataframe by original index and split, then reset index for convenience
            self.df = df_full.loc[df_full["split"] == self.split].loc[indices].reset_index(drop=True)
            self._return_original_idx = True  # Flag to indicate we need to return original index
            # Store the original indices (before reset) for __getitem__ to return
            self._original_indices_map = df_full.loc[df_full["split"] == self.split].loc[indices].index.to_numpy()
        else:
            # This is for the labeled training/validation/test datasets
            self.df = df_full[df_full["split"] == self.split]
            if status is not None:
                self.df = self.df[self.df["status"] == status]
            self.df = self.df.reset_index(drop=True)
            self._return_original_idx = False

    def __len__(self):
        return len(self.df)

    def _resolve(self, p):
        p = Path(p)
        if p.is_absolute():
            return p
        if self.project_root is None:
            return p
        return self.project_root / p  # TODO what is p? There's another variable in the project with the same name, change it

    def __getitem__(self, idx):
        # idx is always a 0-based positional index in the current self.df
        r = self.df.iloc[idx]

        img_path = self._resolve(r["img_path"])
        img = np.load(img_path).astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(0)

        if self._return_original_idx:
            # For pooling dataset, return the original index from the full manifest
            original_manifest_idx = self._original_indices_map[idx]
            return original_manifest_idx, img
        else:
            # For labeled dataset, return image and mask
            mask_path = self._resolve(r["mask_path"])
            mask = np.load(mask_path).astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0)
            return img, mask


def build_datasets(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        ROOT: str | Path  # TODO refactor ROOT variable (what even is it? Fix docstring)
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    This function builds the train and validation datasets.

    Args:
        df_train (pd.DataFrame): dataframe containing training data.
        df_val (pd.DataFrame): dataframe containing validation data.
        ROOT (string | Path): path to ??
    Returns:
        tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: the training and validation datasets.
    """
    train_ds = BaselineDataset(df_train, root=ROOT)
    val_ds = BaselineDataset(df_val, root=ROOT)
    return train_ds, val_ds


def build_dataloaders(
        train_ds: torch.utils.data.Dataset,
        val_ds: torch.utils.data.Dataset,
        df_train: pd.DataFrame,
        oversample: bool,
        k: float
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    This function builds the training and validation dataloaders.

    Args:
        train_ds (torch.utils.data.Dataset): the training dataset.
        val_ds (torch.utils.data.Dataset): the validation dataset.
        df_train (pd.DataFrame): dataframe containing training data.
        oversample (bool): whether to oversample the labels.
        k (float): the oversampling multiplier used by WeightedRandomSampler.
    Returns:
        tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: the training and validation dataloaders.
    """
    # Generate a numpy array of ints corresponding to classes of samples in the train split
    # Note: label[i] corresponds to Dataset item i, because Dataset resets indexes
    labels = df_train["class"].to_numpy().astype(np.int64)

    # Get the number of positive and negative samples
    n_pos = labels.sum()
    assert n_pos > 0
    n_neg = len(labels) - n_pos

    if oversample:
        # Set weights
        w_pos = n_neg / n_pos * k  # Note: a weight of N means that sample has N times the chance of being picked by the loader
        w_neg = 1.0

        # Create array of weights per index of labels array
        weights = np.where(labels == 1, w_pos, w_neg).astype(np.float64)  # int NumPy array

        # Initialize sampler
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights),
            num_samples=len(weights),
            replacement=True
        )
        # Oversampling training dataloader
        train_loader = DataLoader(
            train_ds,
            batch_size=8,
            sampler=sampler,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
    else:
        # Standard, non-oversampling training dataloader
        train_loader = DataLoader(
            train_ds,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )

    # Validation dataloader
    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    return train_loader, val_loader