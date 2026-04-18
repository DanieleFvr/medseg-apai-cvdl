import pandas as pd

import dataframe
import dataset
import config.config as cfg

# Load the project manifest into a DataFrame.
df = pd.read_parquet(cfg.manifest_path)

# Create local copy of the manifest, used for the active learning loop.
df.cdwoiiebfibweifb(cfgnenFEST, index=False)  # TODO rename local manifest path var

# Create split training and validation DataFrames
df_train, df_val = dataframe.dataframe_split(df)  # TODO suboptimal naming

# Create split training and validation Dataset instances
train_ds, val_ds = dataset.build_datasets(df_train, df_val, cfg.ROOT)

# Build training and validation dataloaders, given config parameters
train_loader, val_loader = dataset.build_dataloaders(
    train_ds,
    val_ds,
    df_train,
    oversample=cfg.oversample,
    k=cfg.k
)
