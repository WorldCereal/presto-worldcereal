import functools
import logging
from typing import Any, Callable, Tuple, Union

import numpy as np
import pandas as pd
import torch
import xarray as xr
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dataops import (
    BANDS_GROUPS_IDX,
    NORMED_BANDS,
    S1_S2_ERA5_SRTM,
    DynamicWorld2020_2021,
)
from .dataset import WorldCerealBase, WorldCerealInferenceDataset
from .masking import BAND_EXPANSION
from .presto import Presto
from .utils import device, prep_dataframe

logger = logging.getLogger(__name__)

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
    _ds = WorldCerealInferenceDataset

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
        eo_data, mask = self._ds._extract_eo_data(inarr)
        flat_latlons = self._ds._extract_latlons(inarr, epsg)
        months = self._ds._extract_months(inarr)
        dynamic_world = np.ones((eo_data.shape[0], eo_data.shape[1])) * (
            DynamicWorld2020_2021.class_amount
        )

        return (
            S1_S2_ERA5_SRTM.normalize(eo_data),
            dynamic_world,
            months,
            flat_latlons,
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

        encodings = np.empty(
            [len(dl.dataset), self.model.encoder.embedding_size],  # type: ignore[arg-type]
            dtype=np.float32,
        )

        with torch.no_grad():

            for i, (x, dw, latlons, month, variable_mask) in enumerate(dl):
                x_f, dw_f, latlons_f, month_f, variable_mask_f = [
                    t.to(device) for t in (x, dw, latlons, month, variable_mask)
                ]

                encodings[i * self.batch_size : i * self.batch_size + self.batch_size, :] = (
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

        return encodings

    def extract_presto_features(self, inarr: xr.DataArray, epsg: int = 4326) -> xr.DataArray:

        eo, dynamic_world, months, latlons, mask = self._create_presto_input(inarr, epsg)
        dl = self._create_dataloader(eo, dynamic_world, months, latlons, mask)

        features = self._get_encodings(dl)
        features = rearrange(features, "(x y) c -> x y c", x=len(inarr.x), y=len(inarr.y))
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
    compile: bool = False,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Extracts features from input data using Presto.

    Args:
        inarr (xr.DataArray or pd.DataFrame): Input data as xarray DataArray or pandas DataFrame.
        presto_url (str): URL to the pretrained Presto model.
        epsg (int) : EPSG code describing the coordinates.
        batch_size (int): Batch size to be used for Presto inference.
        compile (bool): Whether to compile the model before extracting features.

    Returns:
        xr.DataArray or np.ndarray: Extracted features as xarray DataArray or numpy ndarray.
    """

    # Load the model
    if presto_url.startswith("http"):
        presto_model = Presto.load_pretrained_url(presto_url=presto_url, strict=False)
    else:
        presto_model = Presto.load_pretrained(model_path=presto_url, strict=False)

    # Compile for optimized inference. Note that warmup takes some time
    # so this is only recommended for larger inference jobs
    if compile:
        presto_model.encoder = compile_encoder(presto_model.encoder)

    presto_extractor = PrestoFeatureExtractor(presto_model, batch_size=batch_size)

    if isinstance(inarr, pd.DataFrame):
        processed_df = process_parquet(inarr)
        test_ds = WorldCerealBase(processed_df)
        dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return presto_extractor._get_encodings(dl)

    elif isinstance(inarr, xr.DataArray):
        return presto_extractor.extract_presto_features(inarr, epsg=epsg)

    else:
        raise ValueError("Input data must be either xr.DataArray or pd.DataFrame")


def process_parquet(df: pd.DataFrame) -> pd.DataFrame:
    # add dummy value + rename stuff for compatibility with existing functions
    df["OPTICAL-B8A"] = 65535
    df.rename(
        columns={
            "S1-SIGMA0-VV": "SAR-VV",
            "S1-SIGMA0-VH": "SAR-VH",
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

    # ----------------------------------------------------------------------------
    # PLACEHOLDER for substituting start_date with one derived from crop calendars
    # df['start_date'] = seasons.get_season_start(df[['lat','lon']])

    # For now, in absence of a relevant start_date, we get time difference with respect
    # to end_date so we can take 12 months counted back from end_date
    df["valid_date_ind"] = (
        (((df["timestamp"] - df["end_date"]).dt.days + 365) / 30).round().astype(int)
    )

    # Now reassign start_date to the actual subset counted back from end_date
    df["start_date"] = df["end_date"] - pd.DateOffset(years=1) + pd.DateOffset(days=1)

    df_pivot = df[(df["valid_date_ind"] >= 0) & (df["valid_date_ind"] < 12)].pivot(
        index=index_columns, columns="valid_date_ind", values=feature_columns
    )

    # ----------------------------------------------------------------------------

    if df_pivot.empty:
        raise ValueError("Left with an empty DataFrame!")

    df_pivot.reset_index(inplace=True)
    df_pivot.columns = [
        f"{xx[0]}-ts{xx[1]}" if isinstance(xx[1], int) else xx[0]
        for xx in df_pivot.columns.to_flat_index()
    ]
    df_pivot.columns = [
        f"{xx}-10m" if any(band in xx for band in bands10m) else xx for xx in df_pivot.columns
    ]
    df_pivot.columns = [
        f"{xx}-20m" if any(band in xx for band in bands20m) else xx for xx in df_pivot.columns
    ]
    df_pivot.columns = [
        f"{xx}-100m" if any(band in xx for band in bands100m) else xx for xx in df_pivot.columns
    ]

    df_pivot["start_date"] = df_pivot["start_date"].dt.date.astype(str)
    df_pivot["end_date"] = df_pivot["end_date"].dt.date.astype(str)
    df_pivot["valid_date"] = df_pivot["valid_date"].dt.date.astype(str)

    df_pivot = prep_dataframe(df_pivot)

    return df_pivot


@functools.lru_cache(maxsize=6)
def compile_encoder(presto_encoder: nn.Module) -> Callable[..., Any]:
    """Helper function that compiles the encoder of a Presto model
    and performs a warm-up on dummy data. The lru_cache decorator
    ensures caching on compute nodes to be able to actually benefit
    from the compilation process.

    Parameters
    ----------
    presto_encoder : nn.Module
        Encoder part of Presto model to compile

    """

    logger.info("Compiling Presto encoder ...")
    presto_encoder = torch.compile(presto_encoder)  # type: ignore

    logger.info("Warming-up ...")
    for _ in range(3):
        presto_encoder(
            torch.rand((1, 12, 17)).to(device),
            torch.ones((1, 12)).to(device).long(),
            torch.rand(1, 2).to(device),
        )

    logger.info("Compilation done.")

    return presto_encoder
