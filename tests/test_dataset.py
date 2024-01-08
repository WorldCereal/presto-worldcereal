from unittest import TestCase

import numpy as np
import pandas as pd
import torch

from presto.dataops import NUM_ORG_BANDS, NUM_TIMESTEPS
from presto.dataset import (
    WorldCerealInferenceDataset,
    WorldCerealLabelledDataset,
    WorldCerealMaskedDataset,
)
from presto.masking import MaskParamsNoDw
from presto.presto import Presto
from presto.utils import data_dir


class TestUtils(TestCase):
    def test_normalize_and_mask(self):
        test_data = np.ones((NUM_TIMESTEPS, NUM_ORG_BANDS))
        test_data[0, 0] = WorldCerealLabelledDataset._NODATAVALUE
        out = WorldCerealLabelledDataset.normalize_and_mask(test_data)
        self.assertEqual(out[0, 0], 0)

    def test_output(self):
        MISSING_DATA_ROW = 0
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

    def test_spatial_dataset(self):
        num_vals = 100
        ds = WorldCerealInferenceDataset()
        # for now, let's just test it runs smoothly
        model = Presto.construct()
        eo, dw, mask, latlons, months, _ = ds[0]
        with torch.no_grad():
            _ = model(
                x=torch.from_numpy(eo).float()[:num_vals],
                dynamic_world=torch.from_numpy(dw).long()[:num_vals],
                latlons=torch.from_numpy(latlons).float()[:num_vals],
                mask=torch.from_numpy(mask).int()[:num_vals],
                month=torch.from_numpy(months).long()[:num_vals],
            )
