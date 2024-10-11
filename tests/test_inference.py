from unittest import TestCase

import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import DataLoader

from presto.dataops import NDVI_INDEX, NODATAVALUE, S2_RGB_INDEX, S2_NIR_10m_INDEX
from presto.dataset import WorldCerealBase, WorldCerealInferenceDataset
from presto.inference import (
    PrestoFeatureExtractor,
    compile_encoder,
    get_presto_features,
)
from presto.presto import Presto
from presto.utils import data_dir, device, prep_dataframe
from tests.utils import read_test_file


class TestInference(TestCase):
    def test_compiled_encoder_without_valid_month_token(self):
        model = Presto.load_pretrained(strict=False, valid_month_as_token=False)
        model.to(device)

        # Check that the encoder has the valid_month_as_token set to False
        assert not model.encoder.valid_month_as_token

        test_data = read_test_file()
        df = prep_dataframe(test_data)
        ds = WorldCerealBase(df)
        dl = DataLoader(
            ds,
            batch_size=512,
            shuffle=False,  # keep as False!
            num_workers=4,
        )

        # First the original uncompiled encoder
        presto_extractor = PrestoFeatureExtractor(model, batch_size=512)
        embeddings = presto_extractor._get_encodings(dl)

        # Now compile the encoder and recompute embeddings
        model.encoder = compile_encoder(model.encoder)
        presto_extractor = PrestoFeatureExtractor(model, batch_size=512)
        embeddings_compiled = presto_extractor._get_encodings(dl)

        # Check that the embeddings are (almost) the same
        np.testing.assert_allclose(embeddings, embeddings_compiled, atol=1e-5)

    def test_compiled_encoder_with_valid_month_token(self):
        model = Presto.load_pretrained(strict=False, valid_month_as_token=True)
        model.to(device)

        # Check that the encoder has the valid_month_as_token set to True
        assert model.encoder.valid_month_as_token

        test_data = read_test_file()
        df = prep_dataframe(test_data)
        ds = WorldCerealBase(df)
        dl = DataLoader(
            ds,
            batch_size=512,
            shuffle=False,  # keep as False!
            num_workers=4,
        )

        # First the original uncompiled encoder
        presto_extractor = PrestoFeatureExtractor(model, batch_size=512, use_valid_date_token=True)
        embeddings = presto_extractor._get_encodings(dl)

        # Now compile the encoder and recompute embeddings
        model.encoder = compile_encoder(model.encoder)
        presto_extractor = PrestoFeatureExtractor(model, batch_size=512, use_valid_date_token=True)
        embeddings_compiled = presto_extractor._get_encodings(dl)

        # Check that the embeddings are (almost) the same
        np.testing.assert_allclose(embeddings, embeddings_compiled, atol=1e-5)

    def test_get_presto_features(self):
        """Test the get_presto_features function. Based on ref features
        generated using this method.
        """
        arr = xr.open_dataarray(data_dir / "test_inference_array.nc")
        arr.attrs["valid_date"] = pd.to_datetime("2020-06-01")

        model_url = str(data_dir / "finetuned_model.pt")
        features_without_valid_month_token = get_presto_features(
            arr, model_url, epsg=32631, use_valid_date_token=False
        )
        features_with_valid_month_token = get_presto_features(
            arr, model_url, epsg=32631, use_valid_date_token=True
        )

        # Uncomment to regenerate ref features
        # features_without_valid_month_token.to_netcdf(
        #   data_dir / "test_inference_features_no_vm.nc"
        # )
        # features_with_valid_month_token.to_netcdf(
        #   data_dir / "test_inference_features_with_vm.nc"
        # )

        # Load ref features
        ref_features_without_valid_month_token = xr.open_dataarray(
            data_dir / "test_inference_features_no_vm.nc"
        )
        ref_features_with_valid_month_token = xr.open_dataarray(
            data_dir / "test_inference_features_with_vm.nc"
        )

        xr.testing.assert_allclose(
            features_without_valid_month_token,
            ref_features_without_valid_month_token,
            rtol=1e-04,
            atol=1e-04,
        )
        xr.testing.assert_allclose(
            features_with_valid_month_token,
            ref_features_with_valid_month_token,
            rtol=1e-04,
            atol=1e-04,
        )
        assert (
            features_without_valid_month_token.dims == ref_features_without_valid_month_token.dims
        )
        assert features_with_valid_month_token.dims == ref_features_with_valid_month_token.dims

    def test_mask_consistency_at_inference(self):
        filepath = (
            data_dir
            / "inference_areas"
            / "WORLDCEREAL-INPUTS-10m_belgium_good_32631_2020-08-01_2022-03-31.nc"
        )
        ds = xr.open_dataset(filepath)
        arr = ds.drop_vars("crs").to_array(dim="bands")
        arr = WorldCerealInferenceDataset._subset_array_temporally(arr)

        # make first 6 timesteps filled with NODATAVALUE
        # expected behavior is that NDVI is masked out
        # at the first six timesteps
        arr.sel(bands="B4").values[:3, :, :] = NODATAVALUE
        arr.sel(bands="B4").values[5, :, :] = NODATAVALUE
        arr.sel(bands="B8").values[3:6, :, :] = NODATAVALUE

        eo, mask = WorldCerealInferenceDataset._extract_eo_data(arr)

        self.assertTrue(set([0, 1, 2, 5]) & set(np.where(mask[:, :, S2_RGB_INDEX])[1]))
        self.assertTrue(set([3, 4, 5]) & set(np.where(mask[:, :, S2_NIR_10m_INDEX])[1]))
        self.assertTrue(mask[:, :6, NDVI_INDEX].all())
