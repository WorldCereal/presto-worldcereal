from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .dataops import BANDS, S1_S2_ERA5_SRTM, DynamicWorld2020_2021
from .masking import MaskedExample, MaskParamsNoDw


class WorldCerealBase(Dataset):
    _NODATAVALUE = 65535
    NUM_TIMESTEPS = 12

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def __len__(self):
        return self.df.shape[0]

    @classmethod
    def row_to_arrays(cls, row: pd.Series) -> Tuple[np.ndarray, np.ndarray, float, int]:
        latlon = np.array([row.lat, row.lon])
        month = datetime.strptime(row.start_date, "%Y-%m-%d").month - 1

        eo_data = np.zeros((cls.NUM_TIMESTEPS, len(BANDS)))
        band_mapping = {
            "OPTICAL-B02-ts{}-10m": "B2",
            "OPTICAL-B03-ts{}-10m": "B3",
            "OPTICAL-B04-ts{}-10m": "B4",
            "OPTICAL-B05-ts{}-20m": "B5",
            "OPTICAL-B06-ts{}-20m": "B6",
            "OPTICAL-B07-ts{}-20m": "B7",
            "OPTICAL-B08-ts{}-10m": "B8",
            "OPTICAL-B8A-ts{}-20m": "B8A",
            "OPTICAL-B11-ts{}-20m": "B11",
            "OPTICAL-B12-ts{}-20m": "B12",
            "SAR-VH-ts{}-20m": "VH",
            "SAR-VV-ts{}-20m": "VV",
            "METEO-precipitation_flux-ts{}-100m": "total_precipitation",
            "METEO-temperature_mean-ts{}-100m": "temperature_2m",
        }
        for df_val, presto_val in band_mapping.items():
            column_names = [df_val.format(t) for t in range(cls.NUM_TIMESTEPS)]
            values = row[column_names].values.astype(float)
            idx_valid = values != cls._NODATAVALUE
            if presto_val in ["VV", "VH"]:
                # convert to dB
                values[idx_valid] = 20 * np.log10(values[idx_valid]) - 83
            elif presto_val == "total_precipitation":
                # scaling, and AgERA5 is in mm, Presto expects m
                values[idx_valid] = values[idx_valid] / (100 * 1000.0)
            elif presto_val == "temperature_2m":
                # remove scaling
                values[idx_valid] = values[idx_valid] / 100
            eo_data[:, BANDS.index(presto_val)] = values
        static_band_mapping = {"DEM-alt-20m": "elevation", "DEM-slo-20m": "slope"}
        for df_val, presto_val in static_band_mapping.items():
            eo_data[:, BANDS.index(presto_val)] = row[df_val]

        return eo_data, latlon, month, row["LANDCOVER_LABEL"] == 11

    def __getitem__(self, idx):
        raise NotImplementedError


class WorldCerealMaskedDataset(WorldCerealBase):
    def __init__(self, dataframe: pd.DataFrame, mask_params: MaskParamsNoDw):
        super().__init__(dataframe)
        self.mask_params = mask_params

    def __getitem__(self, idx):
        # Get the sample
        row = self.df.iloc[idx, :]
        eo, latlon, month, _ = self.row_to_arrays(row)

        # this is copied over from dataops. Sorry
        keep_indices = [idx for idx, val in enumerate(BANDS) if val != "B9"]
        normed_eo = S1_S2_ERA5_SRTM.normalize(eo)
        # TODO: fix this. For now, we replicate the previous behaviour
        normed_eo = np.where(eo[:, keep_indices] == self._NODATAVALUE, normed_eo, 0)
        mask_eo, x_eo, y_eo, strat = self.mask_params.mask_data(normed_eo)

        dynamic_world = np.ones(self.NUM_TIMESTEPS) * (DynamicWorld2020_2021.class_amount)
        mask_dw = np.full(self.NUM_TIMESTEPS, True)
        y_dw = dynamic_world.detach().clone()
        return MaskedExample(
            mask_eo,
            mask_dw,
            x_eo,
            y_eo,
            dynamic_world,
            y_dw,
            month,
            latlon,
            strat,
        )
