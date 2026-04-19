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