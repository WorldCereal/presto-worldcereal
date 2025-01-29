from unittest import TestCase

import numpy as np
import pandas as pd

from presto.dataops import MIN_EDGE_BUFFER, NODATAVALUE, NUM_TIMESTEPS
from presto.input_data_processors import process_parquet


class TestProcessParquet(TestCase):
    def setUp(self):
        # Sample DataFrame setup
        self.start_date = pd.to_datetime("2020-10-01")
        self.n_months = 20

        # normal case
        sample_1_data = {
            "sample_id": ["sample_1"] * self.n_months,
            "timestamp": pd.date_range(
                start=self.start_date,
                end=self.start_date + pd.DateOffset(months=(self.n_months - 1)),
                freq="MS",
            ),
            "start_date": [self.start_date] * self.n_months,
            "valid_date": [self.start_date + pd.DateOffset(months=9)] * self.n_months,
            "DEM-alt-20m": [np.random.randint(1000, size=1)[0]] * self.n_months,
            "DEM-slo-20m": [np.random.randint(1000, size=1)[0]] * self.n_months,
            "S1-SIGMA0-VV": np.random.randint(1000, size=self.n_months),
            "S1-SIGMA0-VH": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B02": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B03": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B04": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B05": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B06": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B07": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B08": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B11": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B12": np.random.randint(1000, size=self.n_months),
            "AGERA5-precipitation-flux": np.random.randint(100, size=self.n_months),
            "AGERA5-temperature-mean": np.random.randint(30, size=self.n_months),
            "CROPTYPE_LABEL": [1200] * self.n_months,
            "lat": [np.random.uniform(-90, 90, size=1)[0]] * self.n_months,
            "lon": [np.random.uniform(-180, 180, size=1)[0]] * self.n_months,
        }

        # valid_date close to start_date
        sample_2_data = {
            "sample_id": ["sample_2"] * self.n_months,
            "timestamp": pd.date_range(
                start=self.start_date,
                end=self.start_date + pd.DateOffset(months=(self.n_months - 1)),
                freq="MS",
            ),
            "start_date": [self.start_date] * self.n_months,
            "valid_date": [self.start_date + pd.DateOffset(months=1)] * self.n_months,
            "DEM-alt-20m": [np.random.randint(1000, size=1)[0]] * self.n_months,
            "DEM-slo-20m": [np.random.randint(1000, size=1)[0]] * self.n_months,
            "S1-SIGMA0-VV": np.random.randint(1000, size=self.n_months),
            "S1-SIGMA0-VH": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B02": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B03": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B04": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B05": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B06": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B07": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B08": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B11": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B12": np.random.randint(1000, size=self.n_months),
            "AGERA5-precipitation-flux": np.random.randint(100, size=self.n_months),
            "AGERA5-temperature-mean": np.random.randint(30, size=self.n_months),
            "CROPTYPE_LABEL": [1310] * self.n_months,
            "lat": [np.random.uniform(-90, 90, size=1)[0]] * self.n_months,
            "lon": [np.random.uniform(-180, 180, size=1)[0]] * self.n_months,
        }

        # valid_date close to end_date
        sample_3_data = {
            "sample_id": ["sample_3"] * self.n_months,
            "timestamp": pd.date_range(
                start=self.start_date,
                end=self.start_date + pd.DateOffset(months=(self.n_months - 1)),
                freq="MS",
            ),
            "start_date": [self.start_date] * self.n_months,
            "valid_date": [self.start_date + pd.DateOffset(months=(self.n_months - 2))]
            * self.n_months,
            "DEM-alt-20m": [np.random.randint(1000, size=1)[0]] * self.n_months,
            "DEM-slo-20m": [np.random.randint(1000, size=1)[0]] * self.n_months,
            "S1-SIGMA0-VV": np.random.randint(1000, size=self.n_months),
            "S1-SIGMA0-VH": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B02": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B03": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B04": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B05": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B06": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B07": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B08": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B11": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B12": np.random.randint(1000, size=self.n_months),
            "AGERA5-precipitation-flux": np.random.randint(100, size=self.n_months),
            "AGERA5-temperature-mean": np.random.randint(30, size=self.n_months),
            "CROPTYPE_LABEL": [1104] * self.n_months,
            "lat": [np.random.uniform(-90, 90, size=1)[0]] * self.n_months,
            "lon": [np.random.uniform(-180, 180, size=1)[0]] * self.n_months,
        }

        # valid_date outside range of extractions
        sample_4_data = {
            "sample_id": ["sample_4"] * self.n_months,
            "timestamp": pd.date_range(
                start=self.start_date,
                end=self.start_date + pd.DateOffset(months=(self.n_months - 1)),
                freq="MS",
            ),
            "start_date": [self.start_date] * self.n_months,
            "valid_date": [self.start_date + pd.DateOffset(months=24)] * self.n_months,
            "DEM-alt-20m": [np.random.randint(1000, size=1)[0]] * self.n_months,
            "DEM-slo-20m": [np.random.randint(1000, size=1)[0]] * self.n_months,
            "S1-SIGMA0-VV": np.random.randint(1000, size=self.n_months),
            "S1-SIGMA0-VH": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B02": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B03": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B04": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B05": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B06": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B07": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B08": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B11": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B12": np.random.randint(1000, size=self.n_months),
            "AGERA5-precipitation-flux": np.random.randint(100, size=self.n_months),
            "AGERA5-temperature-mean": np.random.randint(30, size=self.n_months),
            "CROPTYPE_LABEL": [1102] * self.n_months,
            "lat": [np.random.uniform(-90, 90, size=1)[0]] * self.n_months,
            "lon": [np.random.uniform(-180, 180, size=1)[0]] * self.n_months,
        }
        self.df = pd.concat(
            (
                pd.DataFrame(sample_1_data),
                pd.DataFrame(sample_2_data),
                pd.DataFrame(sample_3_data),
                pd.DataFrame(sample_4_data),
            )
        )
        self.df = self.df.fillna(NODATAVALUE).reset_index(drop=True)

    def test_process_parquet_valid_input(self):
        result = process_parquet(
            self.df,
            ts_freq="month",
            use_valid_time=True,
            required_min_timesteps=NUM_TIMESTEPS,
            min_edge_buffer=MIN_EDGE_BUFFER,
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        self.assertIn("OPTICAL-B02-ts0-10m", result.columns)
        self.assertIn("SAR-VV-ts0-20m", result.columns)
        self.assertIn("METEO-precipitation_flux-ts0-100m", result.columns)

    def test_process_parquet_missing_timestamps(self):
        bands10m = ["OPTICAL-B02", "OPTICAL-B03", "OPTICAL-B04", "OPTICAL-B08"]
        bands20m = [
            "SAR-VH",
            "SAR-VV",
            "OPTICAL-B05",
            "OPTICAL-B06",
            "OPTICAL-B07",
            "OPTICAL-B11",
            "OPTICAL-B12",
            "OPTICAL-B8A",
        ]
        bands100m = ["METEO-precipitation_flux", "METEO-temperature_mean"]
        feature_columns = bands10m + bands20m + bands100m

        # Remove n timestamps to create missing timestamps scenario
        # Make sure not to remove first or last timestamp for each sample
        n = 5
        rows_to_remove = self.df[
            (self.df["timestamp"] != self.start_date)
            & (self.df["timestamp"] != self.start_date + pd.DateOffset(months=self.n_months - 1))
        ].sample(n)

        df_missing = self.df.drop(rows_to_remove.index)
        result = process_parquet(
            df_missing,
            ts_freq="month",
            use_valid_time=True,
            required_min_timesteps=NUM_TIMESTEPS,
            min_edge_buffer=MIN_EDGE_BUFFER,
            return_after_fill=True,
        )
        self.assertIsInstance(result, pd.DataFrame)
        for ii, trow in rows_to_remove.iterrows():
            row_to_check = result[
                (result["sample_id"] == trow["sample_id"])
                & (result["timestamp"] == trow["timestamp"])
            ]
            self.assertFalse(row_to_check.empty)
            self.assertTrue((row_to_check[feature_columns] == NODATAVALUE).values.all())

    def test_process_parquet_valid_date_close_to_start(self):
        result = process_parquet(
            self.df,
            ts_freq="month",
            use_valid_time=True,
            required_min_timesteps=NUM_TIMESTEPS,
            min_edge_buffer=MIN_EDGE_BUFFER,
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)

        expected_start_date = pd.to_datetime(self.start_date) - pd.DateOffset(
            months=(MIN_EDGE_BUFFER - 1)
        )
        obtained_start_date = pd.to_datetime(result.loc["sample_2", "start_date"])
        self.assertTrue(obtained_start_date == expected_start_date)

        expected_available_timesteps = self.n_months + MIN_EDGE_BUFFER - 1
        obtained_available_timesteps = result.loc["sample_2", "available_timesteps"]
        self.assertTrue(obtained_available_timesteps == expected_available_timesteps)

    def test_process_parquet_valid_date_close_to_end(self):
        result = process_parquet(
            self.df,
            ts_freq="month",
            use_valid_time=True,
            required_min_timesteps=NUM_TIMESTEPS,
            min_edge_buffer=MIN_EDGE_BUFFER,
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)

        expected_last_timestep = pd.to_datetime(
            self.start_date + pd.DateOffset(months=((self.n_months - 1) + MIN_EDGE_BUFFER - 1))
        )
        obtained_last_timestep = pd.to_datetime(result.loc["sample_3", "end_date"])
        self.assertTrue(obtained_last_timestep == expected_last_timestep)

        expected_available_timesteps = self.n_months + (MIN_EDGE_BUFFER - 1)
        obtained_available_timesteps = result.loc["sample_3", "available_timesteps"]
        self.assertTrue(obtained_available_timesteps == expected_available_timesteps)

    def test_process_parquet_invalid_input(self):
        result = process_parquet(
            self.df,
            ts_freq="month",
            use_valid_time=True,
            required_min_timesteps=NUM_TIMESTEPS,
            min_edge_buffer=MIN_EDGE_BUFFER,
        )
        self.assertFalse("sample_4" in result.index.unique())

    def test_empty_dataframe(self):
        """Test processing empty dataframe"""
        empty_df = pd.DataFrame(columns=self.df.columns)
        with self.assertRaises(ValueError):
            process_parquet(empty_df)

    def test_invalid_timestamp_freq(self):
        """Test invalid timestamp frequency"""
        with self.assertRaises(ValueError):
            process_parquet(self.df, ts_freq="weekly")

    def test_all_zero_sar_values(self):
        """Test handling of all zero SAR values"""
        test_df = self.df.copy()
        test_df["S1-SIGMA0-VV"] = 0.0
        test_df["S1-SIGMA0-VH"] = 0.0

        result = process_parquet(test_df)
        self.assertTrue((result["SAR-VV-ts0-20m"] == NODATAVALUE).all())
        self.assertTrue((result["SAR-VH-ts0-20m"] == NODATAVALUE).all())

    def test_missing_required_columns(self):
        """Test missing required columns"""
        test_df = self.df.drop(columns=["lat"])
        with self.assertRaises(AttributeError):
            process_parquet(test_df)

    def test_minimum_timesteps_requirement(self):
        """Test minimum timesteps requirement"""
        with self.assertRaises(ValueError):
            process_parquet(self.df, required_min_timesteps=self.n_months + 5)

    def test_valid_position_calculation(self):
        """Test valid position calculation"""
        result = process_parquet(self.df)
        self.assertIn("valid_position", result.columns)
        self.assertTrue((result["valid_position"] >= 0).all())

    def test_feature_columns_initialization(self):
        """Test initialization of missing feature columns"""
        test_df = self.df.drop(columns=["S2-L2A-B08"])
        result = process_parquet(test_df)
        self.assertIn("OPTICAL-B08-ts0-10m", result.columns)
        self.assertTrue((result["OPTICAL-B08-ts0-10m"] == NODATAVALUE).all())

    def test_band_suffix_addition(self):
        """Test correct band suffix addition"""
        result = process_parquet(self.df)

        # Check 10m bands
        self.assertTrue(any(col.endswith("10m") for col in result.columns))
        # Check 20m bands
        self.assertTrue(any(col.endswith("20m") for col in result.columns))
        # Check 100m bands
        self.assertTrue(any(col.endswith("100m") for col in result.columns))

    def test_non_month_start_timestamps(self):
        """Test handling of non-month-start timestamps"""
        test_df = self.df.copy()
        test_df["timestamp"] = test_df["timestamp"] + pd.Timedelta(days=15)

        with self.assertRaises(ValueError):
            process_parquet(test_df)

    def test_date_conversions(self):
        """Test date format conversions"""
        result = process_parquet(self.df)

        # Check date string formats
        self.assertTrue(all(isinstance(d, str) for d in result["start_date"]))
        self.assertTrue(all(isinstance(d, str) for d in result["end_date"]))
        self.assertTrue(all(isinstance(d, str) for d in result["valid_time"]))
