from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .dataops import NUM_TIMESTEPS
from .masking import MaskedExample, MaskParamsNoDw
from .utils import construct_single_presto_input


class WorldCerealDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, mask_params: Optional[MaskParamsNoDw] = None):
        self.df = dataframe
        self.mask_params = mask_params

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        # Get the sample
        row = self.df.iloc[idx, :]

        # Convert inputs and return
        return self.convert_inputs(row)

    def convert_inputs(self, row):

        # Latitude/Longitude
        latlon = torch.tensor([row.lat, row.lon]).float()

        # Month
        month = datetime.strptime(row.start_date, "%Y-%m-%d").month - 1
        month = torch.tensor(month).long()

        # Sentinel-2
        s2_band_mapping = {
            "B02": "B2",
            "B03": "B3",
            "B04": "B4",
            "B05": "B5",
            "B06": "B6",
            "B07": "B7",
            "B08": "B8",
            "B8A": "B8A",
            "B11": "B11",
            "B12": "B12",
        }

        single_sample = pd.DataFrame(
            index=[
                "ts0",
                "ts1",
                "ts2",
                "ts3",
                "ts4",
                "ts5",
                "ts6",
                "ts7",
                "ts8",
                "ts9",
                "ts10",
                "ts11",
            ]
        )
        for b in s2_band_mapping.keys():
            fts = [x for x in self.df.columns if b in x and "ts" in x]
            single_sample[b] = row[fts].values.astype(int)

        s2_data = torch.from_numpy(single_sample.values).float()

        # Sentinel-1
        s1bands = ["VV", "VH"]
        single_sample = pd.DataFrame(
            index=[
                "ts0",
                "ts1",
                "ts2",
                "ts3",
                "ts4",
                "ts5",
                "ts6",
                "ts7",
                "ts8",
                "ts9",
                "ts10",
                "ts11",
            ]
        )
        for b in s1bands:
            fts = [x for x in self.df.columns if b in x and "ts" in x]
            values = row[fts].values.astype(float)
            idx_valid = values != 0
            values[idx_valid] = 20 * np.log10(values[idx_valid]) - 83
            # values[~idx_valid] = -999 # Custom nodata
            # # Don't do this as long as we can't update the mask
            single_sample[b] = values

        s1_data = torch.from_numpy(single_sample.values).float()

        # METEO
        single_sample = pd.DataFrame(
            index=[
                "ts0",
                "ts1",
                "ts2",
                "ts3",
                "ts4",
                "ts5",
                "ts6",
                "ts7",
                "ts8",
                "ts9",
                "ts10",
                "ts11",
            ]
        )
        meteo_band_mapping = {
            "temperature_mean": "temperature_2m",
            "precipitation_flux": "total_precipitation",
        }
        for b in meteo_band_mapping.keys():
            fts = [x for x in self.df.columns if b in x and "ts" in x]
            values = row[fts].values.astype(float)
            idx_valid = values != 0
            values = values / 100.0

            if b == "precipitation_flux":
                values = values / 1000.0
            # idx_valid[~idx_valid] = -999 # Custom nodata # Custom nodata
            # # Don't do this as long as we can't update the mask
            single_sample[b] = values

        meteo_data = torch.from_numpy(single_sample.values).float()

        # Alt/slope
        srtm_data = np.repeat(
            row[["DEM-alt-20m", "DEM-slo-20m"]].values.reshape((1, 2)), 12, axis=0
        )
        srtm_data = torch.from_numpy(srtm_data.astype(float)).float()

        # Construct normalized Presto inputs
        # NOTE: using this method because it conveniently takes care
        # of all the required normalizations
        # the `mask` is not going to be used in training mode
        x, mask, dynamic_world = construct_single_presto_input(
            s2=s2_data,
            s2_bands=list(s2_band_mapping.values()),
            s1=s1_data,
            s1_bands=s1bands,
            era5=meteo_data,
            era5_bands=list(meteo_band_mapping.values()),
            srtm=srtm_data,
            srtm_bands=["elevation", "slope"],
        )

        """
        Adjusting the mask cannot be done for now, as Presto code
        requires all elements in the batch to be masked equally
        cfr. https://github.com/nasaharvest/presto/issues/26#issuecomment-1777120102

        # --------------------------------------------------------
        # Adjust the mask so nodata values are dealt with properly
        #
        # Columns in x:
        # 0, 1: Sentinel-1
        # 2-11: Sentinel-2
        # 12-13: T/Pr
        # 14-15: DEM/Slope
        # 16: NDVI
        # --------------------------------------------------------
        # Sentinel-2: make sure automatically computed NDVI is masked too
        if 0 in s2_data:
            idx_mask = torch.where(s2_data == 0)
            mask[idx_mask[0], idx_mask[1] + 2] = 1
            mask[torch.where(s2_data[:, 0] == 0)[0], -1] = 1  # Manually set NDVI mask

        # Sentinel-1:
        if -999 in s1_data:
            mask[torch.where(s1_data == -999)] = 1

        # Meteo
        if -999 in meteo_data:
            mask[torch.where(meteo_data == -999)] = 1
        """

        if self.mask_params is None:
            # return in the format we can encode it with Presto
            return x.float(), mask.bool(), dynamic_world.long(), latlon, month
        else:
            # return it the way it's done for Presto training
            mask_dw = np.full(NUM_TIMESTEPS, True)
            y_dw = dynamic_world.detach().clone()
            mask_eo, x_eo, y_eo, strat = self.mask_params.mask_data(x.numpy())
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
