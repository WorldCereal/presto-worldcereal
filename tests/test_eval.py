import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np
import rioxarray
import xarray as xr
from presto.eval import WorldCerealEval
from presto.presto import Presto
from presto.utils import data_dir
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

        self.assertTrue("PrestoFineTuningModel" in output["downstream_model_type"].unique())
        self.assertTrue("CatBoostClassifier" in output["downstream_model_type"].unique())
        self.assertEqual(output["f1-score"].isna().sum(), 0)

    # def test_eval_croptype(self):
    #     # loading not strict so that absent valid_month
    #     # in pre-trained model is not a problem
    #     model = Presto.load_pretrained(strict=False)

    #     test_data = read_test_file()
    #     eval_task = WorldCerealEval(
    #     test_data,
    #     test_data,
    #     task_type="cropland",
    #     num_outputs=9,
    #     finetune_classes="CROPTYPE0",
    #     downstream_classes="CROPTYPE9",
    #     dekadal=False,
    #     balance=True,
    # )

    #     output, _ = eval_task.finetuning_results(model, ["CatBoostClassifier"])
    #     # * 283 per model: WorldCereal CatBoost, Presto finetuned, Presto + CatBoost3
    #     self.assertEqual(len(output), 282 * 3)
    #     self.assertTrue("WorldCerealCropland_CatBoostClassifier_f1" in output)
    #     self.assertTrue("WorldCerealCropland_CatBoostClassifier_f1" in output)

    def test_spatial_inference_cropland(
        self,
    ):
        # loading not strict so that absent valid_month
        # in pre-trained model is not a problem
        model = Presto.load_pretrained(strict=False)

        test_data = read_test_file()
        spatial_data_prefix = "belgium_good_2020-12-01_2021-11-30"
        # print(data_dir / f"inference_areas/{spatial_data_prefix}.nc")
        spatial_data = xr.load_dataset(data_dir / f"inference_areas/{spatial_data_prefix}.nc")
        # spatial_data = rioxarray.open_rasterio(
        #     data_dir / f"inference_areas/{spatial_data_prefix}.nc", decode_times=False
        # )
        ground_truth_one_timestep = spatial_data.worldcereal_cropland.values[0, :, :]
        with tempfile.TemporaryDirectory() as tmpdirname:
            eval_task = WorldCerealEval(
                test_data,
                test_data,
                task_type="cropland",
                dekadal=False,
                balance=True,
                spatial_inference_savedir=Path(tmpdirname),
            )
            finetuned_model = eval_task._construct_finetuning_model(model)
            eval_task.spatial_inference(finetuned_model, None)
            # output = xr.open_dataset(
            output = xr.load_dataset(
                Path(tmpdirname) / f"{eval_task.name}_{spatial_data_prefix}_finetuning.nc"
            )
            # np.flip because of how lat lons are stored vs x y
            self.assertTrue(
                np.equal(np.flip(output.ground_truth.values, 0), ground_truth_one_timestep).all()
            )
            self.assertTrue(np.max(output.ndvi) <= 1)
            self.assertTrue(np.max(output.ndvi) >= 0)
