from unittest import TestCase

import pandas as pd

from presto.eval import WorldCerealEval
from presto.presto import Presto
from presto.utils import data_dir


class TestUtils(TestCase):
    def test_eval(self):
        model = Presto.load_pretrained()

        test_data = pd.read_parquet(data_dir / "worldcereal_testdf.parquet")[:20]
        labels = [99] * len(test_data)  # 99 = No cropland
        labels[:10] = [11] * 10  # 11 = Annual cropland
        test_data["LANDCOVER_LABEL"] = labels
        eval_task = WorldCerealEval(test_data, test_data)

        output, _ = eval_task.finetuning_results(model, ["Regression"])
        self.assertEqual(len(output), 564)
        self.assertTrue("WorldCerealCropland_LogisticRegression_f1" in output)
