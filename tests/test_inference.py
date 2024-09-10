from unittest import TestCase

import numpy as np
import xarray as xr
from torch.utils.data import DataLoader

from presto.dataset import WorldCerealBase
from presto.inference import (
    PrestoFeatureExtractor,
    compile_encoder,
    get_presto_features,
)
from presto.presto import Presto
from presto.utils import data_dir, prep_dataframe
from tests.utils import read_test_file


class TestInference(TestCase):
    def test_compiled_encoder(self):
        model = Presto.load_pretrained()
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

    def test_get_presto_features(self):
        """Test the get_presto_features function. Based on ref features
        generated using this method.
        """
        arr = xr.open_dataarray(data_dir / "test_inference_array.nc")
        model_url = str(data_dir / "finetuned_model.pt")
        features = get_presto_features(arr, model_url, epsg=32631)

        # Uncomment to regenerate ref features
        # features.to_netcdf(data_dir / "test_inference_features.nc")

        # Load ref features
        ref_features = xr.open_dataarray(data_dir / "test_inference_features.nc")

        xr.testing.assert_allclose(features, ref_features)
        assert features.dims == ref_features.dims
