from unittest import TestCase

import numpy as np

from src.dataops import NUM_ORG_BANDS, NUM_TIMESTEPS
from src.dataset import WorldCerealLabelledDataset


class TestUtils(TestCase):
    def test_normalize_and_mask(self):
        test_data = np.ones((NUM_TIMESTEPS, NUM_ORG_BANDS))
        test_data[0, 0] = WorldCerealLabelledDataset._NODATAVALUE
        out = WorldCerealLabelledDataset.normalize_and_mask(test_data)
        self.assertEqual(out[0, 0], 0)
