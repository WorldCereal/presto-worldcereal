from unittest import TestCase

import numpy as np
import torch

from presto.dataops import NUM_ORG_BANDS, NUM_TIMESTEPS
from presto.dataset import (
    WorldCerealInferenceDataset,
    WorldCerealLabelledDataset,
    WorldCerealMaskedDataset,
    target_maize,
)
from presto.masking import MaskParamsNoDw
from presto.presto import Presto
from tests.utils import NUM_CROP_POINTS, NUM_MAIZE_POINTS, read_test_file


class TestDataset(TestCase):
    def test_normalize_and_mask(self):
        test_data = np.ones((NUM_TIMESTEPS, NUM_ORG_BANDS))
        test_data[0, 0] = WorldCerealLabelledDataset._NODATAVALUE
        out = WorldCerealLabelledDataset.normalize_and_mask(test_data)
        self.assertEqual(out[0, 0], 0)

    def test_output(self):
        MISSING_DATA_ROW = 0
        df = read_test_file()

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
        eo, dw, mask, flat_latlons, months, _, _, _ = ds[0]
        with torch.no_grad():
            _ = model(
                x=torch.from_numpy(eo).float()[:num_vals],
                dynamic_world=torch.from_numpy(dw).long()[:num_vals],
                latlons=torch.from_numpy(flat_latlons).float()[:num_vals],
                mask=torch.from_numpy(mask).int()[:num_vals],
                month=torch.from_numpy(months).long()[:num_vals],
            )

    def test_combine_predictions(self):
        # adapted from https://github.com/nasaharvest/openmapflow/blob/main/tests/test_inference.py
        flat_lat = np.array([14.95313164, 14.95313165])
        flat_lon = np.array([-86.25070894, -86.25061911])
        batch_predictions = np.array([[0.43200156], [0.55286014], [0.5265], [0.5236109]])
        ndvi = np.array([0.43200156, 0.55286014, 0.5265, 0.5236109])
        worldcereal_labels = np.array([[1, 1], [0, 0], [1, 1], [0, 0]])
        da_predictions = WorldCerealInferenceDataset.combine_predictions(
            all_preds=batch_predictions,
            gt=worldcereal_labels,
            ndvi=ndvi,
            x_coord=flat_lat,
            y_coord=flat_lon,
        )
        df_predictions = da_predictions.to_dataset(dim="bands").to_dataframe()

        # Check size
        self.assertEqual(df_predictions.index.levels[0].name, "y")
        self.assertEqual(df_predictions.index.levels[1].name, "x")
        self.assertEqual(len(df_predictions.index.levels[0]), 2)
        self.assertEqual(len(df_predictions.index.levels[1]), 2)

        # Check coords
        self.assertTrue((df_predictions.index.levels[0].values == flat_lon).all())
        self.assertTrue((df_predictions.index.levels[1].values == flat_lat).all())

        # Check all predictions between 0 and 1
        self.assertTrue(df_predictions["prediction_0"].min() >= 0)
        self.assertTrue(df_predictions["prediction_0"].max() <= 1)
        # Check all ndvi values between 0 and 1
        self.assertTrue(df_predictions["ndvi"].min() >= 0)
        self.assertTrue(df_predictions["ndvi"].max() <= 1)

        # check all the worldcereal labels are 0 or 1
        self.assertTrue(df_predictions["ground_truth"].isin([0, 1]).all())

    def test_targets_correctly_calculated_crop_noncrop(self):
        df = read_test_file()
        ds = WorldCerealLabelledDataset(df)
        num_positives = 0
        for i in range(len(ds)):
            batch = ds[i]
            y = batch[1]
            assert y in [0, 1]
            num_positives += y == 1
        self.assertTrue(num_positives == NUM_CROP_POINTS)

    def test_targets_correctly_calculated_maize(self):
        df = read_test_file()
        ds = WorldCerealLabelledDataset(df, target_function=target_maize)
        num_positives = 0
        for i in range(len(ds)):
            batch = ds[i]
            y = batch[1]
            assert y in [0, 1]
            num_positives += y == 1
        self.assertTrue(num_positives == NUM_MAIZE_POINTS)

    def test_list_correctly_resized(self):
        input_list = [1] * 10
        output_list = WorldCerealLabelledDataset.multiply_list_length_by_float(input_list, 2.5)
        self.assertEqual(len(output_list), 25)
