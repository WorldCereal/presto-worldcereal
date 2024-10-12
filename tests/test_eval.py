import json
import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np
import xarray as xr

from presto.dataset import filter_remove_noncrops
from presto.eval import MIN_SAMPLES_PER_CLASS, Hyperparams, WorldCerealEval
from presto.presto import Presto
from presto.utils import config_dir, data_dir, device
from tests.utils import read_test_file


class TestEval(TestCase):
    def test_eval_cropland(self):
        # loading not strict so that absent valid_month
        # in pre-trained model is not a problem
        model = Presto.load_pretrained(strict=False, valid_month_as_token=False)
        model.to(device)
        # Check that the encoder has the valid_month_as_token set to False
        assert not model.encoder.valid_month_as_token

        test_data = read_test_file()
        eval_task = WorldCerealEval(
            test_data,
            test_data,
            task_type="cropland",
            dekadal=False,
            balance=True,
        )

        hyperparams = Hyperparams()
        hyperparams.max_epochs = 1
        hyperparams.num_workers = 4
        hyperparams.catboost_iterations = 100
        output, _, _ = eval_task.finetuning_results(
            model, ["CatBoostClassifier"], hyperparams=hyperparams
        )

        self.assertTrue("PrestoFineTuningModel" in output["downstream_model_type"].unique())
        self.assertTrue("CatBoostClassifier" in output["downstream_model_type"].unique())
        self.assertTrue(
            ("crop" in output["class"].unique()) and ("not_crop" in output["class"].unique())
        )
        self.assertEqual(
            ((output["support"] >= MIN_SAMPLES_PER_CLASS) & (output["f1-score"].isna())).sum(), 0
        )

    def test_eval_croptype_without_valid_month_token(self):
        path_to_config = config_dir / "default.json"
        with open(path_to_config) as file:
            model_kwargs = json.load(file)

        model = Presto.construct(**model_kwargs)
        model.encoder.valid_month_as_token = False

        finetune_classes = "CROPTYPE0"
        downstream_classes = "CROPTYPE9"
        test_data = read_test_file(finetune_classes, downstream_classes)
        test_data = filter_remove_noncrops(test_data)

        eval_task = WorldCerealEval(
            test_data,
            test_data,
            task_type="croptype",
            finetune_classes=finetune_classes,
            downstream_classes=downstream_classes,
            dekadal=False,
            balance=False,
            augment=False,
            train_masking=0.0,
            use_valid_month=False,
        )

        hyperparams = Hyperparams()
        hyperparams.max_epochs = 1
        hyperparams.num_workers = 4
        hyperparams.catboost_iterations = 100
        output, _, _ = eval_task.finetuning_results(
            model, ["CatBoostClassifier"], hyperparams=hyperparams
        )

        self.assertTrue("PrestoFineTuningModel" in output["downstream_model_type"].unique())
        self.assertTrue("CatBoostClassifier" in output["downstream_model_type"].unique())
        self.assertTrue(
            output["class"][
                ~output["class"].isin(["accuracy", "macro avg", "weighted avg"])
            ].nunique()
            > 2
        )
        self.assertEqual(
            ((output["support"] >= MIN_SAMPLES_PER_CLASS) & (output["f1-score"].isna())).sum(), 0
        )

    def test_eval_croptype_with_valid_month_token(self):
        path_to_config = config_dir / "default.json"
        with open(path_to_config) as file:
            model_kwargs = json.load(file)

        model = Presto.construct(**model_kwargs)
        model.encoder.valid_month_as_token = True

        finetune_classes = "CROPTYPE0"
        downstream_classes = "CROPTYPE9"
        test_data = read_test_file(finetune_classes, downstream_classes)
        test_data = filter_remove_noncrops(test_data)

        eval_task = WorldCerealEval(
            test_data,
            test_data,
            task_type="croptype",
            finetune_classes=finetune_classes,
            downstream_classes=downstream_classes,
            dekadal=False,
            balance=False,
            augment=False,
            train_masking=0.0,
            use_valid_month=True,
        )

        hyperparams = Hyperparams()
        hyperparams.max_epochs = 1
        hyperparams.num_workers = 4
        hyperparams.catboost_iterations = 100
        output, _, _ = eval_task.finetuning_results(
            model, ["CatBoostClassifier"], hyperparams=hyperparams
        )

        self.assertTrue("PrestoFineTuningModel" in output["downstream_model_type"].unique())
        self.assertTrue("CatBoostClassifier" in output["downstream_model_type"].unique())
        self.assertTrue(
            output["class"][
                ~output["class"].isin(["accuracy", "macro avg", "weighted avg"])
            ].nunique()
            > 2
        )
        self.assertEqual(
            ((output["support"] >= MIN_SAMPLES_PER_CLASS) & (output["f1-score"].isna())).sum(), 0
        )

    def test_spatial_inference_croptype_without_valid_month_token(
        self,
    ):
        path_to_config = config_dir / "default.json"
        with open(path_to_config) as file:
            model_kwargs = json.load(file)
        model = Presto.construct(**model_kwargs)
        model.encoder.valid_month_as_token = False

        finetune_classes = "CROPTYPE0"
        downstream_classes = "CROPTYPE9"
        test_data = read_test_file(finetune_classes, downstream_classes)
        test_data = filter_remove_noncrops(test_data)

        spatial_data_prefix = "WORLDCEREAL-INPUTS-10m_belgium_good_32631_2020-08-30_2022-03-03"
        spatial_data = xr.load_dataset(data_dir / f"inference_areas/{spatial_data_prefix}.nc")

        ground_truth_one_timestep = spatial_data.WORLDCEREAL_TEMPORARYCROPS_2021.values

        with tempfile.TemporaryDirectory() as tmpdirname:
            eval_task = WorldCerealEval(
                test_data,
                test_data,
                task_type="croptype",
                finetune_classes=finetune_classes,
                downstream_classes=downstream_classes,
                dekadal=False,
                balance=False,
                use_valid_month=False,
                spatial_inference_savedir=Path(tmpdirname),
            )
            finetuned_model = eval_task._construct_finetuning_model(model)
            eval_task.spatial_inference(finetuned_model, None)
            output = xr.load_dataset(
                Path(tmpdirname) / f"{eval_task.name}_{spatial_data_prefix}_finetuning_croptype.nc"
            )
            gt_values = output.sel(bands="ground_truth")["__xarray_dataarray_variable__"].values
            ndvi_values = output.sel(bands="ndvi")["__xarray_dataarray_variable__"].values
            self.assertTrue(np.equal(gt_values, ground_truth_one_timestep).all())
            self.assertTrue(np.max(ndvi_values) <= 1)
            self.assertTrue(np.max(ndvi_values) >= 0)

    def test_spatial_inference_croptype_with_valid_month_token(
        self,
    ):
        path_to_config = config_dir / "default.json"
        with open(path_to_config) as file:
            model_kwargs = json.load(file)
        model = Presto.construct(**model_kwargs)
        model.encoder.valid_month_as_token = True

        finetune_classes = "CROPTYPE0"
        downstream_classes = "CROPTYPE9"
        test_data = read_test_file(finetune_classes, downstream_classes)
        test_data = filter_remove_noncrops(test_data)

        spatial_data_prefix = "WORLDCEREAL-INPUTS-10m_belgium_good_32631_2020-08-01_2022-03-31"
        spatial_data = xr.load_dataset(data_dir / f"inference_areas/{spatial_data_prefix}.nc")

        ground_truth_one_timestep = spatial_data.WORLDCEREAL_TEMPORARYCROPS_2021.values

        with tempfile.TemporaryDirectory() as tmpdirname:
            eval_task = WorldCerealEval(
                test_data,
                test_data,
                task_type="croptype",
                finetune_classes=finetune_classes,
                downstream_classes=downstream_classes,
                dekadal=False,
                balance=False,
                use_valid_month=True,
                spatial_inference_savedir=Path(tmpdirname),
            )
            finetuned_model = eval_task._construct_finetuning_model(model)
            eval_task.spatial_inference(finetuned_model, None)
            output = xr.load_dataset(
                Path(tmpdirname) / f"{eval_task.name}_{spatial_data_prefix}_finetuning_croptype.nc"
            )
            gt_values = output.sel(bands="ground_truth")["__xarray_dataarray_variable__"].values
            ndvi_values = output.sel(bands="ndvi")["__xarray_dataarray_variable__"].values
            self.assertTrue(np.equal(gt_values, ground_truth_one_timestep).all())
            self.assertTrue(np.max(ndvi_values) <= 1)
            self.assertTrue(np.max(ndvi_values) >= 0)
