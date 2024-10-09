# import tempfile
# from pathlib import Path
# from datetime import datetime
from unittest import TestCase

import numpy as np
import pandas as pd

from presto.dataops import MIN_EDGE_BUFFER, NODATAVALUE
from presto.utils import process_parquet

# from presto.eval import WorldCerealEval
# from presto.presto import Presto
# from presto.utils import data_dir
# from tests.utils import read_test_file


class TestProcessParquet(TestCase):
    def setUp(self):
        # Sample DataFrame setup
        start_date = pd.to_datetime("2020-10-12")
        n_months = 20

        # normal case
        sample_1_data = {
            "sample_id": ["sample_1"] * n_months,
            "timestamp": pd.date_range(
                start=start_date,
                end=start_date + pd.DateOffset(months=n_months),
                freq="m",
            ),
            "start_date": [start_date] * n_months,
            "valid_date": [start_date + pd.DateOffset(months=9)] * n_months,
            "S1-SIGMA0-VV": np.random.randint(1000, size=n_months),
            "S1-SIGMA0-VH": np.random.randint(1000, size=n_months),
            "S2-L2A-B02": np.random.randint(1000, size=n_months),
            "S2-L2A-B03": np.random.randint(1000, size=n_months),
            "S2-L2A-B04": np.random.randint(1000, size=n_months),
            "S2-L2A-B05": np.random.randint(1000, size=n_months),
            "S2-L2A-B06": np.random.randint(1000, size=n_months),
            "S2-L2A-B07": np.random.randint(1000, size=n_months),
            "S2-L2A-B08": np.random.randint(1000, size=n_months),
            "S2-L2A-B11": np.random.randint(1000, size=n_months),
            "S2-L2A-B12": np.random.randint(1000, size=n_months),
            "AGERA5-precipitation-flux": np.random.randint(100, size=n_months),
            "AGERA5-temperature-mean": np.random.randint(30, size=n_months),
        }

        # valid_date close to start_date
        sample_2_data = {
            "sample_id": ["sample_2"] * n_months,
            "timestamp": pd.date_range(
                start=start_date,
                end=start_date + pd.DateOffset(months=n_months),
                freq="m",
            ),
            "start_date": [start_date] * n_months,
            "valid_date": [start_date + pd.DateOffset(months=1)] * n_months,
            "S1-SIGMA0-VV": np.random.randint(1000, size=n_months),
            "S1-SIGMA0-VH": np.random.randint(1000, size=n_months),
            "S2-L2A-B02": np.random.randint(1000, size=n_months),
            "S2-L2A-B03": np.random.randint(1000, size=n_months),
            "S2-L2A-B04": np.random.randint(1000, size=n_months),
            "S2-L2A-B05": np.random.randint(1000, size=n_months),
            "S2-L2A-B06": np.random.randint(1000, size=n_months),
            "S2-L2A-B07": np.random.randint(1000, size=n_months),
            "S2-L2A-B08": np.random.randint(1000, size=n_months),
            "S2-L2A-B11": np.random.randint(1000, size=n_months),
            "S2-L2A-B12": np.random.randint(1000, size=n_months),
            "AGERA5-precipitation-flux": np.random.randint(100, size=n_months),
            "AGERA5-temperature-mean": np.random.randint(30, size=n_months),
        }

        # valid_date close to end_date
        sample_3_data = {
            "sample_id": ["sample_3"] * n_months,
            "timestamp": pd.date_range(
                start=start_date,
                end=start_date + pd.DateOffset(months=n_months),
                freq="m",
            ),
            "start_date": [start_date] * n_months,
            "valid_date": [start_date + pd.DateOffset(months=18)] * n_months,
            "S1-SIGMA0-VV": np.random.randint(1000, size=n_months),
            "S1-SIGMA0-VH": np.random.randint(1000, size=n_months),
            "S2-L2A-B02": np.random.randint(1000, size=n_months),
            "S2-L2A-B03": np.random.randint(1000, size=n_months),
            "S2-L2A-B04": np.random.randint(1000, size=n_months),
            "S2-L2A-B05": np.random.randint(1000, size=n_months),
            "S2-L2A-B06": np.random.randint(1000, size=n_months),
            "S2-L2A-B07": np.random.randint(1000, size=n_months),
            "S2-L2A-B08": np.random.randint(1000, size=n_months),
            "S2-L2A-B11": np.random.randint(1000, size=n_months),
            "S2-L2A-B12": np.random.randint(1000, size=n_months),
            "AGERA5-precipitation-flux": np.random.randint(100, size=n_months),
            "AGERA5-temperature-mean": np.random.randint(30, size=n_months),
        }

        # valid_date outside range of extractions
        sample_4_data = {
            "sample_id": ["sample_4"] * n_months,
            "timestamp": pd.date_range(
                start=start_date,
                end=start_date + pd.DateOffset(months=n_months),
                freq="m",
            ),
            "start_date": [start_date] * n_months,
            "valid_date": [start_date + pd.DateOffset(months=24)] * n_months,
            "S1-SIGMA0-VV": np.random.randint(1000, size=n_months),
            "S1-SIGMA0-VH": np.random.randint(1000, size=n_months),
            "S2-L2A-B02": np.random.randint(1000, size=n_months),
            "S2-L2A-B03": np.random.randint(1000, size=n_months),
            "S2-L2A-B04": np.random.randint(1000, size=n_months),
            "S2-L2A-B05": np.random.randint(1000, size=n_months),
            "S2-L2A-B06": np.random.randint(1000, size=n_months),
            "S2-L2A-B07": np.random.randint(1000, size=n_months),
            "S2-L2A-B08": np.random.randint(1000, size=n_months),
            "S2-L2A-B11": np.random.randint(1000, size=n_months),
            "S2-L2A-B12": np.random.randint(1000, size=n_months),
            "AGERA5-precipitation-flux": np.random.randint(100, size=n_months),
            "AGERA5-temperature-mean": np.random.randint(30, size=n_months),
        }

        self.df = pd.concat(
            (
                pd.DataFrame(sample_1_data),
                pd.DataFrame(sample_2_data),
                pd.DataFrame(sample_3_data),
                pd.DataFrame(sample_4_data),
            )
        )
        self.df = self.df.fillna(NODATAVALUE)

    def test_process_parquet_valid_input(self):
        result = process_parquet(self.df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        self.assertIn("OPTICAL-B02-ts0-10m", result.columns)
        self.assertIn("SAR-VV-ts0-20m", result.columns)
        self.assertIn("METEO-precipitation_flux-ts0-100m", result.columns)

    def test_process_parquet_missing_timestamps(self):
        # Remove some timestamps to create missing timestamps scenario
        df_missing = self.df.drop(self.df.index[1])
        result = process_parquet(df_missing)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        self.assertIn("OPTICAL-B02-ts0-10m", result.columns)
        self.assertIn(
            "OPTICAL-B02-ts1-10m", result.columns
        )  # Check if missing timestamp was added

    def test_process_parquet_valid_date_close_to_start(self):
        result = process_parquet(self.df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)

        expected_start_date = pd.to_datetime("2020-10-12") - pd.DateOffset(
            months=(MIN_EDGE_BUFFER - 1)
        )
        obtained_start_date = result[result["sample_id"] == "sample_2"]["start_date"].iloc[0]
        self.assertTrue(obtained_start_date == expected_start_date)

        initial_available_timesteps = 20
        expected_available_timesteps = initial_available_timesteps + (MIN_EDGE_BUFFER - 1)
        obtained_available_timesteps = result[result["sample_id"] == "sample_2"][
            "available_timesteps"
        ].iloc[0]
        self.assertTrue(obtained_available_timesteps == expected_available_timesteps)

    def test_process_parquet_valid_date_close_to_end(self):
        result = process_parquet(self.df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)

        initial_available_timesteps = 20

        expected_last_timestep = pd.to_datetime("2020-10-12") + pd.DateOffset(
            months=(initial_available_timesteps + MIN_EDGE_BUFFER - 1)
        )
        obtained_last_timestep = result[result["sample_id"] == "sample_3"]["timestep"].max()
        self.assertTrue(obtained_last_timestep == expected_last_timestep)

        expected_available_timesteps = initial_available_timesteps + (MIN_EDGE_BUFFER - 1)
        obtained_available_timesteps = result[result["sample_id"] == "sample_3"][
            "available_timesteps"
        ].iloc[0]
        self.assertTrue(obtained_available_timesteps == expected_available_timesteps)

    def test_process_parquet_invalid_input(self):
        result = process_parquet(self.df)
        self.assertFalse("sample_4" in result["sample_id"].unique())
