from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .dataops import (
    BANDS,
    BANDS_GROUPS_IDX,
    NORMED_BANDS,
    S1_S2_ERA5_SRTM,
    DynamicWorld2020_2021,
)
from .masking import BAND_EXPANSION, MaskedExample, MaskParamsNoDw

IDX_TO_BAND_GROUPS = {}
for band_group_idx, (key, val) in enumerate(BANDS_GROUPS_IDX.items()):
    for idx in val:
        IDX_TO_BAND_GROUPS[NORMED_BANDS[idx]] = band_group_idx


class WorldCerealBase(Dataset):
    _NODATAVALUE = 65535
    NUM_TIMESTEPS = 12
    BAND_MAPPING = {
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
    STATIC_BAND_MAPPING = {"DEM-alt-20m": "elevation", "DEM-slo-20m": "slope"}

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def __len__(self):
        return self.df.shape[0]

    @classmethod
    def row_to_arrays(
        cls, row: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        # https://stackoverflow.com/questions/45783891/is-there-a-way-to-speed-up-the-pandas-getitem-getitem-axis-and-get-label
        # This is faster than indexing the series every time!
        row_d = pd.Series.to_dict(row)

        latlon = np.array([row_d["lat"], row_d["lon"]], dtype=np.float32)
        month = datetime.strptime(row_d["start_date"], "%Y-%m-%d").month - 1

        eo_data = np.zeros((cls.NUM_TIMESTEPS, len(BANDS)))
        # an assumption we make here is that all timesteps for a token
        # have the same masking
        mask_per_token = np.zeros((cls.NUM_TIMESTEPS, len(BANDS_GROUPS_IDX)))
        for df_val, presto_val in cls.BAND_MAPPING.items():
            values = np.array([float(row_d[df_val.format(t)]) for t in range(cls.NUM_TIMESTEPS)])
            idx_valid = values != cls._NODATAVALUE
            mask_per_token[:, IDX_TO_BAND_GROUPS[presto_val]] = np.clip(
                mask_per_token[:, IDX_TO_BAND_GROUPS[presto_val]] + (~idx_valid), a_min=0, a_max=1
            )
            if presto_val in ["VV", "VH"]:
                # convert to dB
                idx_valid = idx_valid & (values > 0)
                values[idx_valid] = 20 * np.log10(values[idx_valid]) - 83
            elif presto_val == "total_precipitation":
                # scaling, and AgERA5 is in mm, Presto expects m
                values[idx_valid] = values[idx_valid] / (100 * 1000.0)
            elif presto_val == "temperature_2m":
                # remove scaling
                values[idx_valid] = values[idx_valid] / 100
            eo_data[:, BANDS.index(presto_val)] = values
        for df_val, presto_val in cls.STATIC_BAND_MAPPING.items():
            eo_data[:, BANDS.index(presto_val)] = row_d[df_val]

        return eo_data, mask_per_token.astype(bool), latlon, month, row_d["LANDCOVER_LABEL"] == 11

    def __getitem__(self, idx):
        raise NotImplementedError

    @classmethod
    def normalize_and_mask(cls, eo: np.ndarray):
        # this is copied over from dataops. Sorry
        keep_indices = [idx for idx, val in enumerate(BANDS) if val != "B9"]
        normed_eo = S1_S2_ERA5_SRTM.normalize(eo)
        # TODO: fix this. For now, we replicate the previous behaviour
        normed_eo = np.where(eo[:, keep_indices] != cls._NODATAVALUE, normed_eo, 0)
        return normed_eo


class WorldCerealMaskedDataset(WorldCerealBase):
    def __init__(self, dataframe: pd.DataFrame, mask_params: MaskParamsNoDw):
        super().__init__(dataframe)
        self.mask_params = mask_params

    def __getitem__(self, idx):
        # Get the sample
        row = self.df.iloc[idx, :]
        eo, real_mask_per_token, latlon, month, _ = self.row_to_arrays(row)
        mask_eo, x_eo, y_eo, strat = self.mask_params.mask_data(
            self.normalize_and_mask(eo), real_mask_per_token
        )
        real_mask_per_variable = np.repeat(real_mask_per_token, BAND_EXPANSION, axis=1)

        dynamic_world = np.ones(self.NUM_TIMESTEPS) * (DynamicWorld2020_2021.class_amount)
        mask_dw = np.full(self.NUM_TIMESTEPS, True)
        y_dw = dynamic_world.copy()
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
            real_mask_per_variable,
        )


class WorldCerealLabelledDataset(WorldCerealBase):
    # 0: no information, 10: could be both annual or perennial
    FILTER_LABELS = [0, 10]

    def __init__(self, dataframe: pd.DataFrame):
        dataframe = dataframe.loc[~dataframe.LANDCOVER_LABEL.isin(self.FILTER_LABELS)]
        super().__init__(dataframe)

    def __getitem__(self, idx):
        # Get the sample
        row = self.df.iloc[idx, :]
        eo, mask_per_token, latlon, month, target = self.row_to_arrays(row)
        mask_per_variable = np.repeat(mask_per_token, BAND_EXPANSION, axis=1)
        num_masked_tokens = sum(sum(mask_per_token))
        return (
            self.normalize_and_mask(eo),
            target,
            np.ones(self.NUM_TIMESTEPS) * (DynamicWorld2020_2021.class_amount),
            latlon,
            month,
            num_masked_tokens,
            mask_per_variable,
        )
