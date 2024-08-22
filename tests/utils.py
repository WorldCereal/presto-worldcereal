import numpy as np
import pandas as pd

from presto.dataset import WorldCerealBase
from presto.utils import data_dir


def read_test_file(
    finetune_classes: str = "CROPTYPE0", downstream_classes: str = "CROPTYPE9"
) -> pd.DataFrame:
    test_df = pd.read_parquet(data_dir / "worldcereal_testdf_upd.parquet")
    # this is to align the parquet file with the new parquet files
    # shared in https://github.com/WorldCereal/presto-worldcereal/pull/34
    test_df.rename(
        {"catboost_prediction": "worldcereal_prediction"},
        axis=1,
        inplace=True,
    )
    test_df.reset_index(inplace=True)
    test_df = WorldCerealBase.map_croptypes(test_df, finetune_classes, downstream_classes)

    return test_df
