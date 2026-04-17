import dataframe
import dataset
import config.config as cfg


# Load the project manifest into a DataFrame.
df = pd.read_parquet(manifest_path)

# Create local copy of the manifest, used for the active learning loop.
df.to_parquet(LOCAL_MANIFEST, index=False)  # TODO rename local manifest path var