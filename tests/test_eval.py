import json
import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np
import rioxarray
import xarray as xr

from presto.dataset import WorldCerealBase, filter_remove_noncrops
from presto.eval import WorldCerealEval
from presto.presto import Presto
from presto.utils import config_dir, data_dir
from tests.utils import read_test_file


class TestEval(TestCase):
    def test_eval_cropland(self):
        # loading not strict so that absent valid_month
        # in pre-trained model is not a problem
        model = Presto.load_pretrained(strict=False)

        test_data = read_test_file()
        eval_task = WorldCerealEval(
            test_data,
            test_data,
            task_type="cropland",
            dekadal=False,
            balance=True,
        )

        output, _, _ = eval_task.finetuning_results(model, ["CatBoostClassifier"])

        print(output.iloc[:10])

        self.assertTrue("PrestoFineTuningModel" in output["downstream_model_type"].unique())
        self.assertTrue("CatBoostClassifier" in output["downstream_model_type"].unique())
        self.assertTrue(
            ("crop" in output["class"].unique()) and ("not_crop" in output["class"].unique())
        )
        self.assertEqual(output["f1-score"].isna().sum(), 0)

    def test_eval_croptype(self):
        path_to_config = config_dir / "default.json"
        with open(path_to_config) as file:
            model_kwargs = json.load(file)
        model = Presto.construct(**model_kwargs)

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
        )

        output, _, _ = eval_task.finetuning_results(model, ["CatBoostClassifier"])

        print(output.iloc[:14])

        self.assertTrue("PrestoFineTuningModel" in output["downstream_model_type"].unique())
        self.assertTrue("CatBoostClassifier" in output["downstream_model_type"].unique())
        self.assertTrue(
            output["class"][~output["class"].isin(["accuracy", "macro avg"])].nunique() > 2
        )
        self.assertEqual(output["f1-score"].isna().sum(), 0)

    def test_spatial_inference_cropland(
        self,
    ):
        # loading not strict so that absent valid_month
        # in pre-trained model is not a problem
        model = Presto.load_pretrained(strict=False)

        test_data = read_test_file()
        spatial_data_prefix = "belgium_good_2020-12-01_2021-11-30"
        spatial_data = xr.load_dataset(data_dir / f"inference_areas/{spatial_data_prefix}.nc")

        ground_truth_one_timestep = spatial_data.worldcereal_cropland.values[0, :, :]
        with tempfile.TemporaryDirectory() as tmpdirname:
            # print(f"tmpdirname: {tmpdirname}")

            # config_dir.parent
            tmpdirname = config_dir.parent / "output" / "tests"
            print(f"tmpdirname: {tmpdirname}")
            # path_to_config = config_dir.parent / "output" / "tests"

            eval_task = WorldCerealEval(
                test_data,
                test_data,
                task_type="cropland",
                dekadal=False,
                balance=True,
                spatial_inference_savedir=Path(tmpdirname),
            )
            finetuned_model = eval_task._construct_finetuning_model(model)

            tpath = Path(tmpdirname) / f"{eval_task.name}_{spatial_data_prefix}_finetuning.nc"
            print(f"path to preds: {tpath}")

            eval_task.spatial_inference(finetuned_model, None)
            output = xr.load_dataset(
                Path(tmpdirname) / f"{eval_task.name}_{spatial_data_prefix}_finetuning.nc"
            )
            # np.flip because of how lat lons are stored vs x y
            self.assertTrue(
                np.equal(np.flip(output.ground_truth.values, 0), ground_truth_one_timestep).all()
            )
            self.assertTrue(np.max(output.ndvi) <= 1)
            self.assertTrue(np.max(output.ndvi) >= 0)

    # def test_spatial_inference_croptype(
    #     self,
    # ):
    #     # loading not strict so that absent valid_month
    #     # in pre-trained model is not a problem
    #     model = Presto.load_pretrained(strict=False)

    #     test_data = read_test_file()
    #     spatial_data_prefix = "belgium_good_2020-12-01_2021-11-30"
    #     spatial_data = xr.load_dataset(data_dir / f"inference_areas/{spatial_data_prefix}.nc")

    #     ground_truth_one_timestep = spatial_data.worldcereal_cropland.values[0, :, :]
    #     with tempfile.TemporaryDirectory() as tmpdirname:
    #         eval_task = WorldCerealEval(
    #             test_data,
    #             test_data,
    #             task_type="cropland",
    #             dekadal=False,
    #             balance=True,
    #             spatial_inference_savedir=Path(tmpdirname),
    #         )
    #         finetuned_model = eval_task._construct_finetuning_model(model)

    #         print(f"path to preds: {Path(tmpdirname) / f"{eval_task.name}_{spatial_data_prefix}_finetuning.nc"}")

    #         eval_task.spatial_inference(finetuned_model, None)
    #         output = xr.load_dataset(
    #             Path(tmpdirname) / f"{eval_task.name}_{spatial_data_prefix}_finetuning.nc"
    #         )
    #         # np.flip because of how lat lons are stored vs x y
    #         self.assertTrue(
    #             np.equal(np.flip(output.ground_truth.values, 0), ground_truth_one_timestep).all()
    #         )
    #         self.assertTrue(np.max(output.ndvi) <= 1)
    #         self.assertTrue(np.max(output.ndvi) >= 0)
