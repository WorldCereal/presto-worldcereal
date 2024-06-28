from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
import xarray as xr
from einops import rearrange
from pyproj import Transformer
from torch.utils.data import DataLoader, TensorDataset

from .dataops import (
    BANDS,
    BANDS_GROUPS_IDX,
    NORMED_BANDS,
    S1_S2_ERA5_SRTM,
    DynamicWorld2020_2021,
)
from .dataset import WorldCerealLabelledDataset
from .eval import WorldCerealEval
from .masking import BAND_EXPANSION
from .presto import Presto
from .utils import device

# Index to band groups mapping
IDX_TO_BAND_GROUPS = {
    NORMED_BANDS[idx]: band_group_idx
    for band_group_idx, (_, val) in enumerate(BANDS_GROUPS_IDX.items())
    for idx in val
}


class PrestoFeatureExtractor:
    def __init__(self, model: Presto, batch_size: int = 8192):
        """
        Initialize the PrestoFeatureExtractor with a Presto model.

        Args:
            model (Presto): The Presto model used for feature extraction.
            batch_size (int): Batch size for dataloader.
        """
        self.model = model
        self.batch_size = batch_size

    _NODATAVALUE = 65535

    BAND_MAPPING = {
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
        "VH": "VH",
        "VV": "VV",
        "precipitation-flux": "total_precipitation",
        "temperature-mean": "temperature_2m",
    }

    @classmethod
    def _preprocess_band_values(
        cls, values: np.ndarray, presto_band: str
    ) -> np.ndarray:
        """
        Preprocesses the band values based on the given presto_val.

        Args:
            values (np.ndarray): Array of band values to preprocess.
            presto_val (str): Name of the band for preprocessing.

        Returns:
            np.ndarray: Preprocessed array of band values.
        """
        if presto_band in ["VV", "VH"]:
            # Convert to dB
            values = 20 * np.log10(values) - 83
        elif presto_band == "total_precipitation":
            # Scale precipitation and convert mm to m
            values = values / (100 * 1000.0)
        elif presto_band == "temperature_2m":
            # Remove scaling
            values = values / 100
        return values

    @classmethod
    def _extract_eo_data(cls, inarr: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts EO data and mask arrays from the input xarray.DataArray.

        Args:
            inarr (xr.DataArray): Input xarray.DataArray containing EO data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing EO data array and mask array.
        """
        num_pixels = len(inarr.x) * len(inarr.y)
        num_timesteps = len(inarr.t)

        eo_data = np.zeros((num_pixels, num_timesteps, len(BANDS)))
        mask = np.zeros((num_pixels, num_timesteps, len(BANDS_GROUPS_IDX)))

        for org_band, presto_band in cls.BAND_MAPPING.items():
            if org_band in inarr.coords["bands"]:
                values = np.swapaxes(
                    inarr.sel(bands=org_band).values.reshape((num_timesteps, -1)), 0, 1
                )
                idx_valid = values != cls._NODATAVALUE
                values = cls._preprocess_band_values(values, presto_band)
                eo_data[:, :, BANDS.index(presto_band)] = values
                mask[:, :, IDX_TO_BAND_GROUPS[presto_band]] += ~idx_valid

        return eo_data, mask

    @staticmethod
    def _extract_latlons(inarr: xr.DataArray, epsg: int) -> np.ndarray:
        """
        Extracts latitudes and longitudes from the input xarray.DataArray.

        Args:
            inarr (xr.DataArray): Input xarray.DataArray containing spatial coordinates.
            epsg (int): EPSG code for coordinate reference system.

        Returns:
            np.ndarray: Array containing extracted latitudes and longitudes.
        """
        # EPSG:4326 is the supported crs for presto
        lon, lat = np.meshgrid(inarr.x, inarr.y)
        transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(lon, lat)
        latlons = rearrange(np.stack([lat, lon]), "c x y -> (x y) c")

        #  2D array where each row represents a pair of latitude and longitude coordinates.
        return latlons

    @staticmethod
    def _extract_months(inarr: xr.DataArray) -> np.ndarray:
        """
        Calculate the start month based on the first timestamp in the input array,
        and create an array of the same length filled with that start month value.

        Parameters:
        - inarr: xarray.DataArray or numpy.ndarray
            Input array containing timestamps.

        Returns:
        - months: numpy.ndarray
            Array of start month values, with the same length as the input array.
        """
        num_instances = len(inarr.x) * len(inarr.y)

        start_month = (
            inarr.t.values[0].astype("datetime64[M]").astype(int) % 12 + 1
        ) - 1

        months = np.ones((num_instances)) * start_month
        return months

    def _create_dataloader(
        self,
        eo: np.ndarray,
        dynamic_world: np.ndarray,
        months: np.ndarray,
        latlons: np.ndarray,
        mask: np.ndarray,
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader for encoding features.

        Args:
            eo_data (np.ndarray): Array containing Earth Observation data.
            dynamic_world (np.ndarray): Array containing dynamic world data.
            latlons (np.ndarray): Array containing latitude and longitude coordinates.
            inarr (xr.DataArray): Input xarray.DataArray.
            mask (np.ndarray): Array containing masking data.

        Returns:
            DataLoader: PyTorch DataLoader for encoding features.
        """

        dl = DataLoader(
            TensorDataset(
                torch.from_numpy(eo).float(),
                torch.from_numpy(dynamic_world).long(),
                torch.from_numpy(latlons).float(),
                torch.from_numpy(months).long(),
                torch.from_numpy(mask).float(),
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )

        return dl

    def _create_presto_input(
        self, inarr: xr.DataArray, epsg: int = 4326
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eo_data, mask = self._extract_eo_data(inarr)
        latlons = self._extract_latlons(inarr, epsg)
        months = self._extract_months(inarr)
        dynamic_world = np.ones((eo_data.shape[0], eo_data.shape[1])) * (
            DynamicWorld2020_2021.class_amount
        )

        return (
            S1_S2_ERA5_SRTM.normalize(eo_data),
            dynamic_world,
            months,
            latlons,
            np.repeat(mask, BAND_EXPANSION, axis=-1),
        )

    def _get_encodings(self, dl: DataLoader) -> np.ndarray:
        """
        Get encodings from DataLoader.

        Args:
            dl (DataLoader): PyTorch DataLoader containing data for encoding.

        Returns:
            np.ndarray: Array containing encoded features.
        """

        all_encodings = []

        for b in dl:
            try:
                x, dw, latlons, month, variable_mask = b
            except ValueError:
                x, _, dw, latlons, month, variable_mask = b

            x_f, dw_f, latlons_f, month_f, variable_mask_f = [
                t.to(device) for t in (x, dw, latlons, month, variable_mask)
            ]

            with torch.no_grad():
                encodings = (
                    self.model.encoder(
                        x_f,
                        dynamic_world=dw_f.long(),
                        mask=variable_mask_f,
                        latlons=latlons_f,
                        month=month_f,
                    )
                    .cpu()
                    .numpy()
                )

            all_encodings.append(encodings)

        return np.concatenate(all_encodings, axis=0)

    def extract_presto_features(
        self, inarr: xr.DataArray, epsg: int = 4326
    ) -> xr.DataArray:
        eo, dynamic_world, months, latlons, mask = self._create_presto_input(
            inarr, epsg
        )
        dl = self._create_dataloader(eo, dynamic_world, months, latlons, mask)

        features = self._get_encodings(dl)
        features = rearrange(
            features, "(x y) c -> x y c", x=len(inarr.x), y=len(inarr.y)
        )
        ft_names = [f"presto_ft_{i}" for i in range(128)]
        features_da = xr.DataArray(
            features,
            dims=["x", "y", "bands"],
            coords={"x": inarr.x, "y": inarr.y, "bands": ft_names},
        )

        return features_da


def get_presto_features(
    inarr: Union[pd.DataFrame, xr.DataArray],
    presto_url: str,
    epsg: int = 4326,
    batch_size: int = 8192,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Extracts features from input data using Presto.

    Args:
        inarr (xr.DataArray or pd.DataFrame): Input data as xarray DataArray or pandas DataFrame.
        presto_url (str): URL to the pretrained Presto model.
        epsg (int) : EPSG code describing the coordinates.
        batch_size (int): Batch size to be used for Presto inference.

    Returns:
        xr.DataArray or np.ndarray: Extracted features as xarray DataArray or numpy ndarray.
    """

    # Load the model
    if presto_url.startswith("http"):
        presto_model = Presto.load_pretrained_url(presto_url=presto_url, strict=False)
    else:
        presto_model = Presto.load_pretrained(model_path=presto_url, strict=False)

    presto_extractor = PrestoFeatureExtractor(presto_model, batch_size=batch_size)

    if type(inarr) == pd.DataFrame:
        processed_df = process_parquet(inarr)
        test_ds = WorldCerealLabelledDataset(processed_df)
        dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        features = presto_extractor._get_encodings(dl)

    if type(inarr) == xr.DataArray:
        features = presto_extractor.extract_presto_features(inarr, epsg=epsg)

    return features


def process_parquet(df: pd.DataFrame) -> pd.DataFrame:
    # add dummy value + rename stuff for compatibility with existing functions
    df["OPTICAL-B8A"] = 0
    df.rename(
        columns={
            "S1-SIGMA0-VV": "SAR-VH",
            "S1-SIGMA0-VH": "SAR-VV",
            "S2-L2A-B02": "OPTICAL-B02",
            "S2-L2A-B03": "OPTICAL-B03",
            "S2-L2A-B04": "OPTICAL-B04",
            "S2-L2A-B05": "OPTICAL-B05",
            "S2-L2A-B06": "OPTICAL-B06",
            "S2-L2A-B07": "OPTICAL-B07",
            "S2-L2A-B08": "OPTICAL-B08",
            "S2-L2A-B11": "OPTICAL-B11",
            "S2-L2A-B12": "OPTICAL-B12",
            "AGERA5-precipitation-flux": "METEO-precipitation_flux",
            "AGERA5-temperature-mean": "METEO-temperature_mean",
        },
        inplace=True,
    )

    feature_columns = [
        "METEO-precipitation_flux",
        "METEO-temperature_mean",
        "SAR-VH",
        "SAR-VV",
        "OPTICAL-B02",
        "OPTICAL-B03",
        "OPTICAL-B04",
        "OPTICAL-B08",
        "OPTICAL-B8A",
        "OPTICAL-B05",
        "OPTICAL-B06",
        "OPTICAL-B07",
        "OPTICAL-B11",
        "OPTICAL-B12",
    ]
    index_columns = [
        "CROPTYPE_LABEL",
        "DEM-alt-20m",
        "DEM-slo-20m",
        "LANDCOVER_LABEL",
        "POTAPOV-LABEL-10m",
        "WORLDCOVER-LABEL-10m",
        "aez_zoneid",
        "end_date",
        "lat",
        "lon",
        "start_date",
        "sample_id",
        "valid_date",
    ]

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

    # PLACEHOLDER for substituting start_date with one derived from crop calendars
    # df['start_date'] = seasons.get_season_start(df[['lat','lon']])

    df["valid_date_ind"] = (
        ((df["timestamp"] - df["start_date"]).dt.days / 30).round().astype(int)
    )

    # once the start date is settled, we take 12 months from that as input to Presto
    df_pivot = df[(df["valid_date_ind"] >= 0) & (df["valid_date_ind"] < 12)].pivot(
        index=index_columns, columns="valid_date_ind", values=feature_columns
    )

    df_pivot.reset_index(inplace=True)
    df_pivot.columns = [
        f"{xx[0]}-ts{xx[1]}" if isinstance(xx[1], int) else xx[0]
        for xx in df_pivot.columns.to_flat_index()
    ]
    df_pivot.columns = [
        f"{xx}-10m" if any(band in xx for band in bands10m) else xx
        for xx in df_pivot.columns
    ]
    df_pivot.columns = [
        f"{xx}-20m" if any(band in xx for band in bands20m) else xx
        for xx in df_pivot.columns
    ]
    df_pivot.columns = [
        f"{xx}-100m" if any(band in xx for band in bands100m) else xx
        for xx in df_pivot.columns
    ]

    df_pivot["start_date"] = df_pivot["start_date"].dt.date.astype(str)
    df_pivot["end_date"] = df_pivot["end_date"].dt.date.astype(str)
    df_pivot["valid_date"] = df_pivot["valid_date"].dt.date.astype(str)

    df_pivot = WorldCerealEval.prep_dataframe(df_pivot)

    return df_pivot
