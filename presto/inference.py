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
from .utils import device, process_parquet

logger = logging.getLogger(__name__)

# Index to band groups mapping
IDX_TO_BAND_GROUPS = {
    NORMED_BANDS[idx]: band_group_idx
    for band_group_idx, (_, val) in enumerate(BANDS_GROUPS_IDX.items())
    for idx in val
}


class PrestoFeatureExtractor:
    def __init__(self, model: Presto, use_valid_date_token: bool = False, batch_size: int = 8192):
        """
        Initialize the PrestoFeatureExtractor with a Presto model.

        Args:
            model (Presto): The Presto model used for feature extraction.
            use_valid_date_token (bool): Use `valid_date` as input token to focus Presto.
            batch_size (int): Batch size for dataloader.
        """
        self.model = model
        self.use_valid_date_token = use_valid_date_token
        self.batch_size = batch_size

        if use_valid_date_token:
            logger.warning('Initializing PrestoFeatureExtractor with "valid_date" token.')
        else:
            logger.warning('Initializing PrestoFeatureExtractor without "valid_date" token.')

    _NODATAVALUE = 65535
    _ds = WorldCerealInferenceDataset

    def _create_dataloader(
        self,
        eo: np.ndarray,
        dynamic_world: np.ndarray,
        months: np.ndarray,
        latlons: np.ndarray,
        mask: np.ndarray,
        valid_months: np.ndarray,
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader for encoding features.

        Args:
            eo_data (np.ndarray): Array containing Earth Observation data.
            dynamic_world (np.ndarray): Array containing dynamic world data.
            latlons (np.ndarray): Array containing latitude and longitude coordinates.
            inarr (xr.DataArray): Input xarray.DataArray.
            mask (np.ndarray): Array containing masking data.
            months (np.ndarray): Array containing month data.
            valid_months (np.ndarray): Array containing valid month data.

        Returns:
            DataLoader: PyTorch DataLoader for encoding features.
        """

        dl = DataLoader(
            TensorDataset(
                torch.from_numpy(eo).float(),
                torch.from_numpy(dynamic_world).long(),
                torch.from_numpy(latlons).float(),
                torch.from_numpy(months).long(),
                torch.from_numpy(valid_months).long(),
                torch.from_numpy(mask).float(),
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )

        return dl

    def _create_presto_input(
        self, inarr: xr.DataArray, epsg: int = 4326
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eo_data, mask = self._ds._extract_eo_data(inarr)
        flat_latlons = self._ds._extract_latlons(inarr, epsg)
        months = self._ds._extract_months(inarr)

        valid_month = pd.to_datetime(inarr.attrs["valid_date"]).month - 1
        valid_months = np.full_like(months, valid_month)
        dynamic_world = np.ones((eo_data.shape[0], eo_data.shape[1])) * (
            DynamicWorld2020_2021.class_amount
        )

        return (
            S1_S2_ERA5_SRTM.normalize(eo_data),
            dynamic_world,
            months,
            flat_latlons,
            np.repeat(mask, BAND_EXPANSION, axis=-1),
            valid_months,
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

            for i, (x, dw, latlons, month, valid_month, variable_mask) in enumerate(dl):
                x_f, dw_f, latlons_f, month_f, valid_month_f, variable_mask_f = [
                    t.to(device) for t in (x, dw, latlons, month, valid_month, variable_mask)
                ]
                encodings[i * self.batch_size : i * self.batch_size + self.batch_size, :] = (
                    self.model.encoder(
                        x_f,
                        dynamic_world=dw_f.long(),
                        mask=variable_mask_f,
                        latlons=latlons_f,
                        month=month_f,
                        valid_month=(valid_month_f if self.use_valid_date_token else None),
                    )
                    .cpu()
                    .numpy()
                )

        return encodings

    def extract_presto_features(self, inarr: xr.DataArray, epsg: int = 4326) -> xr.DataArray:

        eo, dynamic_world, months, latlons, mask, valid_months = self._create_presto_input(
            inarr, epsg
        )
        dl = self._create_dataloader(eo, dynamic_world, months, latlons, mask, valid_months)

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
    use_valid_date_token: bool = False,
    batch_size: int = 8192,
    compile: bool = False,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Extracts features from input data using Presto.

    Args:
        inarr (xr.DataArray or pd.DataFrame): Input data as xarray DataArray or pandas DataFrame.
        presto_url (str): URL to the pretrained Presto model.
        epsg (int) : EPSG code describing the coordinates.
        use_valid_date_token (bool) : Use `valid_date` as input token to focus Presto.
        batch_size (int): Batch size to be used for Presto inference.
        compile (bool): Whether to compile the model before extracting features.

    Returns:
        xr.DataArray or np.ndarray: Extracted features as xarray DataArray or numpy ndarray.
    """

    # Load the model
    from_url = presto_url.startswith("http")
    presto_model = Presto.load_pretrained(model_path=presto_url, from_url=from_url, strict=False)

    # Compile for optimized inference. Note that warmup takes some time
    # so this is only recommended for larger inference jobs
    if compile:
        presto_model.encoder = compile_encoder(presto_model.encoder)

    presto_extractor = PrestoFeatureExtractor(
        presto_model, use_valid_date_token=use_valid_date_token, batch_size=batch_size
    )

    if isinstance(inarr, pd.DataFrame):
        processed_df = process_parquet(inarr)
        test_ds = WorldCerealBase(processed_df)
        dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return presto_extractor._get_encodings(dl)

    elif isinstance(inarr, xr.DataArray):
        # Check if we have the expected 12 timesteps
        if len(inarr.t) != 12:
            raise ValueError(f"Can only run Presto on 12 timesteps, got: {len(inarr.t)}")
        return presto_extractor.extract_presto_features(inarr, epsg=epsg)

    else:
        raise ValueError("Input data must be either xr.DataArray or pd.DataFrame")


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
            x=torch.rand((1, 12, 17)).to(device),
            dynamic_world=torch.ones((1, 12)).to(device).long(),
            latlons=torch.rand(1, 2).to(device),
            valid_month=torch.tensor([5]).to(device)
            if presto_encoder.valid_month_as_token
            else None,
        )

    logger.info("Compilation done.")

    return presto_encoder
