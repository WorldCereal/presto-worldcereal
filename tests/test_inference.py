from unittest import TestCase

import numpy as np
from torch.utils.data import DataLoader

from presto.dataset import WorldCerealBase
from presto.inference import PrestoFeatureExtractor, compile_encoder
from presto.presto import Presto
from presto.utils import prep_dataframe
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
