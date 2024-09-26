import json
from unittest import TestCase

import numpy as np
import torch
from presto.dataops import NODATAVALUE, NUM_ORG_BANDS, NUM_TIMESTEPS
from presto.dataset import (WorldCerealInferenceDataset,
                            WorldCerealLabelledDataset,
                            WorldCerealMaskedDataset, filter_remove_noncrops)
from presto.eval import Hyperparams, WorldCerealEval
from presto.masking import MaskParamsNoDw
from presto.presto import Presto
from presto.utils import config_dir, device
from tests.utils import read_test_file


class TestDataset(TestCase):
    def test_normalize_and_mask(self):
        test_data = np.ones((NUM_TIMESTEPS, NUM_ORG_BANDS))
        test_data[0, 0] = NODATAVALUE
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
        # self.assertTrue((mask[true_mask] == True).all())
        self.assertTrue((mask[true_mask] == 1).all())
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
        path_to_config = config_dir / "default.json"
        with open(path_to_config) as file:
            model_kwargs = json.load(file)

        model = Presto.construct(**model_kwargs)
        model.to(device)
        eo, dw, mask, latlons, months, _, valid_months, _, _ = ds[0]

        with torch.no_grad():
            _ = model(
                x=torch.from_numpy(eo).float()[:num_vals],
                dynamic_world=torch.from_numpy(dw).long()[:num_vals],
                latlons=torch.from_numpy(latlons).float()[:num_vals],
                mask=torch.from_numpy(mask).int()[:num_vals],
                month=torch.from_numpy(months).long()[:num_vals],
                valid_month=torch.from_numpy(valid_months).long()[:num_vals],
            )

    def test_combine_predictions(self):
        # copied from https://github.com/nasaharvest/openmapflow/blob/main/tests/test_inference.py
        x_coord = np.array([14.95313164, 14.95323164, 14.95333164, 14.95343164, 14.95353164])
        y_coord = np.array([-86.25070894, -86.25061911, -86.25052928, -86.25043945, -86.25034962])

        b2 = np.random.rand(len(y_coord) * len(x_coord))
        b3 = np.random.rand(len(y_coord) * len(x_coord))
        b4 = np.random.rand(len(y_coord) * len(x_coord))
        ndvi = np.random.rand(len(y_coord) * len(x_coord))

        worldcereal_labels = np.random.randint(
            low=0, high=2, size=(len(y_coord) * len(x_coord)), dtype=int
        )
        batch_predictions = np.random.rand(len(y_coord) * len(x_coord))
        all_preds_ewoc_code = np.full_like(batch_predictions, 110000000)

        da_predictions = WorldCerealInferenceDataset.combine_predictions(
            all_preds=batch_predictions,
            gt=worldcereal_labels,
            ndvi=ndvi,
            all_preds_ewoc_code=all_preds_ewoc_code,
            all_probs=batch_predictions,
            b2=b2,
            b3=b3,
            b4=b4,
            x_coord=x_coord,
            y_coord=y_coord,
        )

        df_predictions = da_predictions.to_dataset(dim="bands").to_dataframe()

        # Check size
        self.assertEqual(df_predictions.index.levels[0].name, "y")
        self.assertEqual(df_predictions.index.levels[1].name, "x")
        self.assertEqual(len(df_predictions.index.levels[0]), len(y_coord))
        self.assertEqual(len(df_predictions.index.levels[1]), len(x_coord))

        # Check coords
        self.assertTrue((df_predictions.index.levels[0].values == y_coord).all())
        self.assertTrue((df_predictions.index.levels[1].values == x_coord).all())

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
        NUM_CROP_POINTS = (df["LANDCOVER_LABEL"] == 11).sum()
        ds = WorldCerealLabelledDataset(df)
        num_positives = 0
        for i in range(len(ds)):
            batch = ds[i]
            y = batch[1]
            assert y in [0, 1]
            num_positives += y == 1
        self.assertTrue(num_positives == NUM_CROP_POINTS)

    def test_list_correctly_resized(self):
        input_list = [1] * 10
        output_list = WorldCerealLabelledDataset.multiply_list_length_by_float(input_list, 2.5)
        self.assertEqual(len(output_list), 25)


    def test_balancing_croptype(self):
        finetune_classes = "CROPTYPE0"
        downstream_classes = "CROPTYPE9"
        test_data = read_test_file(finetune_classes, downstream_classes)
        test_data = filter_remove_noncrops(test_data)

        balance_indices = eval_task_balance.ds_class(
            eval_task_balance.train_df,
            task_type=eval_task_balance.task_type,
            croptype_list=eval_task_balance.croptype_list,
            balance=eval_task_balance.balance,
            augment=eval_task_balance.augment,
            mask_ratio=eval_task_balance.train_masking,
        ).indices
        class_counts_balanced = (
            eval_task_balance.train_df.finetune_class.iloc[balance_indices]
            .value_counts()
            .sort_index()
        )
        class_counts_df = pd.DataFrame(class_counts_balanced)
        class_counts_df.columns = ["balanced_counts"]

        imbalance_indices = eval_task_imbalance.ds_class(
            eval_task_imbalance.train_df,
            task_type=eval_task_imbalance.task_type,
            croptype_list=eval_task_imbalance.croptype_list,
            balance=eval_task_imbalance.balance,
            augment=eval_task_imbalance.augment,
            mask_ratio=eval_task_imbalance.train_masking,
        ).indices
        class_counts_imbalanced = (
            eval_task_imbalance.train_df.finetune_class.iloc[imbalance_indices]
            .value_counts()
        )
        class_counts_df["imbalanced_counts"] = class_counts_df.index.map(class_counts_imbalanced)

        self.assertTrue((class_counts_df["balanced_counts"] >= class_counts_df["imbalanced_counts"]).all())

    def test_augment_temporal_jittering(self):
        row_d = {"available_timesteps": 1000, "valid_position": 20}

        valid_position = int(row_d["valid_position"])

        timestep_positions_base = WorldCerealLabelledDataset.get_timestep_positions(
            row_d, augment=False, is_ssl=False
        )
        timestep_positions_augment1 = WorldCerealLabelledDataset.get_timestep_positions(
            row_d, augment=True, is_ssl=False
        )
        timestep_positions_augment2 = WorldCerealLabelledDataset.get_timestep_positions(
            row_d, augment=True, is_ssl=False
        )

        timestep_positions_ssl1 = WorldCerealLabelledDataset.get_timestep_positions(
            row_d, augment=False, is_ssl=True
        )
        timestep_positions_ssl2 = WorldCerealLabelledDataset.get_timestep_positions(
            row_d, augment=False, is_ssl=True
        )
        
        self.assertTrue(timestep_positions_base[0] == (valid_position - WorldCerealBase.NUM_TIMESTEPS // 2)) 
        self.assertTrue(timestep_positions_base[-1] == (valid_position + (WorldCerealBase.NUM_TIMESTEPS // 2) - 1))
        self.assertTrue(timestep_positions_augment1[0] != timestep_positions_augment2[0])
        self.assertTrue(timestep_positions_ssl1[0] != timestep_positions_ssl2[0])
        self.asserTrue((valid_position not in timestep_positions_ssl1) or (valid_position not in timestep_positions_ssl2))