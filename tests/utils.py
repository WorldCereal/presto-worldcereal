import pandas as pd

from presto.dataset import WorldCerealBase
from presto.input_data_processors import process_parquet
from presto.utils import data_dir


def read_test_file(
    finetune_classes: str = "CROPTYPE0", downstream_classes: str = "CROPTYPE9"
) -> pd.DataFrame:
    test_parquet_fpath = data_dir / "test_long_parquet_2017_CAN_AAFC-ACIGTD.parquet"
    test_df_long = pd.read_parquet(test_parquet_fpath, engine="fastparquet")
    print(test_df_long["timestamp"].min(), test_df_long["timestamp"].max())
    test_df = process_parquet(test_df_long)
    test_df.reset_index(inplace=True)

    test_df.rename(
        {"WORLDCOVER-LABEL-10m": "worldcereal_prediction"},
        axis=1,
        inplace=True,
    )
    test_df.reset_index(inplace=True)
    test_df = WorldCerealBase.map_croptypes(test_df, finetune_classes, downstream_classes)

    return test_df
