import pandas as pd
from pathlib import Path


def manifest_load(manifest_path: str | Path) -> pd.DataFrame:
    # TODO refactor original manifest variable: not clear which manifest
    """
    This function loads the original manifest file.

    Args:
        manifest_path (Path): Path to the manifest file.
    """
    return pd.read_parquet(manifest_path)


def manifest_local_copy(
        manifest_path: str | Path,
        dst_path: str | Path
) -> None:
    """
    This function creates a local copy of the manifest file.

    Args:
        manifest_path (Path): Path to the original manifest file.
        dst_path (Path): Path to the manifest file copy.
    """
    df = pd.read_parquet(manifest_path)
    df.to_parquet(dst_path, index=False)


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