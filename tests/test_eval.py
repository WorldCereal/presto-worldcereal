from unittest import TestCase

import pandas as pd

from src.eval import WorldCerealEval
from src.presto import Presto
from src.utils import data_dir


class TestUtils(TestCase):
    def test_eval(self):
        model = Presto.load_pretrained()

        test_data = pd.read_parquet(data_dir / "worldcereal_testdf.parquet")[:20]
        labels = [99] * len(test_data)  # 99 = No cropland
        labels[:10] = [11] * 10  # 11 = Annual cropland
        test_data["LANDCOVER_LABEL"] = labels
        eval_task = WorldCerealEval(test_data, test_data)

        output = eval_task.finetuning_results(model, ["Regression"])
        self.assertEqual(len(output), 180)
        self.assertTrue("WorldCerealCropland_LogisticRegression_f1" in output)
