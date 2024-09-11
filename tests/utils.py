from glob import glob

import numpy as np
import pandas as pd

from presto.utils import process_parquet

# we will have a different number of maize labels for easier testing
NUM_CROP_POINTS = 10
NUM_MAIZE_POINTS = 20


def read_test_file() -> pd.DataFrame:
    parquet_file = "/vitodata/worldcereal/features/preprocessedinputs-monthly-nointerp/\
worldcereal_training_data.parquet"
    files = sorted(glob(f"{parquet_file}/**/*.parquet"))
    f = files[7]
    test_df_long = pd.read_parquet(f, engine="fastparquet")
    test_df = process_parquet(test_df_long)
    test_df.reset_index(inplace=True)

    test_df.rename(
        {"WORLDCOVER-LABEL-10m": "worldcereal_prediction"},
        axis=1,
        inplace=True,
    )
    test_df["sample_id"] = np.arange(len(test_df))
    test_df["year"] = 2021
    labels = [99] * len(test_df)  # 99 = No cropland
    labels[:NUM_CROP_POINTS] = [11] * NUM_CROP_POINTS  # 11 = Annual cropland
    test_df["LANDCOVER_LABEL"] = labels

    croptype_labels = [99] * len(test_df)
    croptype_labels[:NUM_MAIZE_POINTS] = [1200] * NUM_MAIZE_POINTS  # 1200 = Maize
    test_df["CROPTYPE_LABEL"] = croptype_labels

    return test_df
