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
    def test_eval(self):
        model = Presto.load_pretrained()

        test_data = read_test_file()
        eval_task = WorldCerealEval(test_data, test_data)

        output, _ = eval_task.finetuning_results(model, ["CatBoostClassifier"])
        # * 283 per model: WorldCereal CatBoost, Presto finetuned, Presto + CatBoost3
        self.assertEqual(len(output), 282 * 3)
        self.assertTrue("WorldCerealCropland_CatBoostClassifier_f1" in output)
        self.assertTrue("WorldCerealCropland_CatBoostClassifier_f1" in output)

    def test_spatial_inference(
        self,
    ):
        model = Presto.load_pretrained()

        test_data = read_test_file()
        spatial_data_prefix = "belgium_good_2020-12-01_2021-11-30"
        spatial_data = rioxarray.open_rasterio(
            data_dir / f"inference_areas/{spatial_data_prefix}.nc", decode_times=False
        )
        ground_truth_one_timestep = spatial_data.worldcereal_cropland.values[0, :, :]
        with tempfile.TemporaryDirectory() as tmpdirname:
            eval_task = WorldCerealEval(
                test_data, test_data, spatial_inference_savedir=Path(tmpdirname)
            )
            finetuned_model = eval_task._construct_finetuning_model(model)
            eval_task.spatial_inference(finetuned_model, None)
            output = xr.open_dataset(
                Path(tmpdirname) / f"{eval_task.name}_{spatial_data_prefix}_finetuning.nc"
            )
            # np.flip because of how lat lons are stored vs x y
            self.assertTrue(
                np.equal(np.flip(output.ground_truth.values, 0), ground_truth_one_timestep).all()
            )
            self.assertTrue(np.max(output.ndvi) <= 1)
            self.assertTrue(np.max(output.ndvi) >= 0)
