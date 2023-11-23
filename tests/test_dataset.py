from unittest import TestCase

import numpy as np
import pandas as pd

from src.dataops import NUM_ORG_BANDS, NUM_TIMESTEPS
from src.dataset import WorldCerealLabelledDataset, WorldCerealMaskedDataset
from src.masking import MaskParamsNoDw
from src.utils import data_dir


class TestUtils(TestCase):
    def test_normalize_and_mask(self):
        test_data = np.ones((NUM_TIMESTEPS, NUM_ORG_BANDS))
        test_data[0, 0] = WorldCerealLabelledDataset._NODATAVALUE
        out = WorldCerealLabelledDataset.normalize_and_mask(test_data)
        self.assertEqual(out[0, 0], 0)

    def test_output(self):
        MISSING_DATA_ROW = 86268
        df = pd.read_parquet(data_dir / "worldcereal_testdf.parquet")
        location_index = df.index.get_loc(MISSING_DATA_ROW)
        strategies = [
            "group_bands",
            "random_timesteps",
            "chunk_timesteps",
            "random_combinations",
        ]
        ds = WorldCerealMaskedDataset(df, MaskParamsNoDw(strategies, 0.8))
        vals = ds[location_index]
        mask = vals[0]
        eo_data = vals[2]
        y = vals[3]
        true_mask = vals[9]
        self.assertTrue((mask[true_mask] == True).all())
        self.assertTrue((eo_data[true_mask] == 0).all())
        # we are very confident these should not be 0 after normalization
        combined_s2_s1 = (eo_data + y)[:, :-5]
        combined_s2_s1_mask = true_mask[:, :-5]
        self.assertTrue((combined_s2_s1[~combined_s2_s1_mask] != 0).all())

        # finally, check the masking we do in train.py works as expected
        mask[true_mask] = False
        s1_s2_mask = mask[:, :-5]
        self.assertTrue((y[:, :-5][s1_s2_mask] != 0).all())
