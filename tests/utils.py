import numpy as np
import pandas as pd

from presto.utils import data_dir

# we will have a different number of maize labels for easier testing
NUM_CROP_POINTS = 10
NUM_MAIZE_POINTS = 20


def read_test_file() -> pd.DataFrame:
    test_df = pd.read_parquet(data_dir / "worldcereal_testdf.parquet")[:20]
    # this is to align the parquet file with the new parquet files
    # shared in https://github.com/WorldCereal/presto-worldcereal/pull/34
    test_df.rename(
        {"catboost_prediction": "worldcereal_prediction"},
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
