import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np
import xarray as xr
from presto.eval import MIN_SAMPLES_PER_CLASS, WorldCerealEval
from presto.presto import Presto
from presto.utils import data_dir, device

from tests.utils import read_test_file


class TestEval(TestCase):
    def test_eval(self):
       # loading not strict so that absent valid_month
        # in pre-trained model is not a problem
        model = Presto.load_pretrained(strict=False)
        model.to(device)

        test_data = read_test_file()
        eval_task = WorldCerealEval(
            test_data,
            test_data,
            dekadal=False,
        )

        output, _ = eval_task.finetuning_results(model, ["CatBoostClassifier"])
        # self.assertTrue("PrestoFineTuningModel" in output["downstream_model_type"].unique())
        # self.assertTrue("CatBoostClassifier" in output["downstream_model_type"].unique())
        # self.assertTrue(
        #     ("crop" in output["class"].unique()) and ("not_crop" in output["class"].unique())
        # )
        # self.assertEqual(
        #     ((output["support"] >= MIN_SAMPLES_PER_CLASS) & (output["f1-score"].isna())).sum(), 0
        # )

def test_spatial_inference(
    self,
):
    model = Presto.load_pretrained()

    test_data = read_test_file()
    spatial_data_prefix = "WORLDCEREAL-INPUTS-10m_belgium_good_32631_2020-08-30_2022-03-03"
    spatial_data = xr.open_dataset(data_dir / f"inference_areas/{spatial_data_prefix}.nc")
    ground_truth_one_timestep = spatial_data.WORLDCEREAL_TEMPORARYCROPS_2021.values

    with tempfile.TemporaryDirectory() as tmpdirname:
        eval_task = WorldCerealEval(
            test_data, test_data, spatial_inference_savedir=Path(tmpdirname)
        )
        finetuned_model = eval_task._construct_finetuning_model(model)
        eval_task.spatial_inference(finetuned_model, None)
        output = xr.open_dataarray(
            Path(tmpdirname) / f"{eval_task.name}_{spatial_data_prefix}_finetuning.nc"
        )
        # np.flip because of how lat lons are stored vs x y
        self.assertTrue(
            np.equal(
                output.sel(bands="ground_truth", drop=True).values,
                ground_truth_one_timestep,
            ).all()
        )
        self.assertTrue(np.max(output.sel(bands="ndvi").values) <= 1)
        self.assertTrue(np.max(output.sel(bands="ndvi").values) >= 0)
