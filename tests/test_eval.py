from unittest import TestCase

import pandas as pd

from src.eval import WorldCerealEval
from src.presto import Presto
from src.utils import data_dir


class TestUtils(TestCase):
    model = Presto.load_pretrained()

    test_data = pd.read_parquet(data_dir / "worldcereal_testdf.parquet")[:20]
    labels = [0] * len(test_data)
    labels[:10] = [11] * 10
    test_data["LANDCOVER_LABEL"] = labels
    eval_task = WorldCerealEval(test_data, test_data)

    output = eval_task.finetuning_results(model, ["Regression"])
    assert len(output) == 3
