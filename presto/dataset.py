import json
import logging
from datetime import datetime, timedelta
from math import modf
from pathlib import Path
from random import sample
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from einops import rearrange
from pyproj import CRS, Transformer
from torch.utils.data import Dataset

from .dataops import (
    BANDS,
    BANDS_GROUPS_IDX,
    NORMED_BANDS,
    S1_S2_ERA5_SRTM,
    DynamicWorld2020_2021,
)
from .masking import BAND_EXPANSION, MaskedExample, MaskParamsNoDw
from .utils import DEFAULT_SEED, data_dir, load_world_df

logger = logging.getLogger("__main__")

IDX_TO_BAND_GROUPS = {}
for band_group_idx, (key, val) in enumerate(BANDS_GROUPS_IDX.items()):
    for idx in val:
        IDX_TO_BAND_GROUPS[NORMED_BANDS[idx]] = band_group_idx

with open(data_dir / "croptype_mappings" / "croptype_classes.json") as f:
    CLASS_MAPPINGS = json.load(f)


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
        cls,
        row: pd.Series,
        task_type: str = "cropland",
        croptype_list: List = [],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        # https://stackoverflow.com/questions/45783891/is-there-a-way-to-speed-up-the-pandas-getitem-getitem-axis-and-get-label
        # This is faster than indexing the series every time!
        row_d = pd.Series.to_dict(row)

        latlon = np.array([row_d["lat"], row_d["lon"]], dtype=np.float32)
        month = datetime.strptime(row_d["start_date"], "%Y-%m-%d").month - 1

        # adding workaround for compatibility between Phase I and Phase II datasets.
        # (in Phase II, the relevant attribute name was changed to valid_time)
        # once we fully move to Phase II data, this should be replaced to valid_tome only.
        if "valid_date" in row_d.keys():
            valid_month = datetime.strptime(row_d["valid_date"], "%Y-%m-%d").month - 1
        elif "valid_time" in row_d.keys():
            valid_month = datetime.strptime(row_d["valid_time"], "%Y-%m-%d").month - 1
        else:
            logger.error("Dataset does not contain neither valid_date, nor valid_time attribute.")

        eo_data = np.zeros((cls.NUM_TIMESTEPS, len(BANDS)))
        # an assumption we make here is that all timesteps for a token
        # have the same masking
        mask = np.zeros((cls.NUM_TIMESTEPS, len(BANDS_GROUPS_IDX)))
        for df_val, presto_val in cls.BAND_MAPPING.items():
            values = np.array([float(row_d[df_val.format(t)]) for t in range(cls.NUM_TIMESTEPS)])
            # this occurs for the DEM values in one point in Fiji
            values = np.nan_to_num(values, nan=cls._NODATAVALUE)
            idx_valid = values != cls._NODATAVALUE
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
            mask[:, IDX_TO_BAND_GROUPS[presto_val]] += ~idx_valid
            eo_data[:, BANDS.index(presto_val)] = values * idx_valid
        for df_val, presto_val in cls.STATIC_BAND_MAPPING.items():
            # this occurs for the DEM values in one point in Fiji
            values = np.nan_to_num(row_d[df_val], nan=cls._NODATAVALUE)
            idx_valid = values != cls._NODATAVALUE
            eo_data[:, BANDS.index(presto_val)] = values * idx_valid
            mask[:, IDX_TO_BAND_GROUPS[presto_val]] += ~idx_valid

        return (cls.check(eo_data), mask.astype(bool), latlon, month, valid_month)

    def __getitem__(self, idx):
        # Get the sample
        row = self.df.iloc[idx, :]
        eo, mask_per_token, latlon, month, valid_month = self.row_to_arrays(row)
        mask_per_variable = np.repeat(mask_per_token, BAND_EXPANSION, axis=1)
        return (
            self.normalize_and_mask(eo),
            np.ones(self.NUM_TIMESTEPS) * (DynamicWorld2020_2021.class_amount),
            latlon,
            month,
            valid_month,
            mask_per_variable,
        )

    @classmethod
    def normalize_and_mask(cls, eo: np.ndarray):
        # TODO: this can be removed
        keep_indices = [idx for idx, val in enumerate(BANDS) if val != "B9"]
        normed_eo = S1_S2_ERA5_SRTM.normalize(eo)
        # TODO: fix this. For now, we replicate the previous behaviour
        normed_eo = np.where(eo[:, keep_indices] != cls._NODATAVALUE, normed_eo, 0)
        return normed_eo

    @staticmethod
    def map_croptypes(
        df: pd.DataFrame,
        finetune_classes="CROPTYPE0",
        downstream_classes="CROPTYPE19",
    ) -> pd.DataFrame:

        wc2ewoc_map = pd.read_csv(data_dir / "croptype_mappings" / "wc2eurocrops_map.csv")
        wc2ewoc_map["ewoc_code"] = wc2ewoc_map["ewoc_code"].str.replace("-", "").astype(int)

        ewoc_map = pd.read_csv(data_dir / "croptype_mappings" / "eurocrops_map_wcr_edition.csv")
        ewoc_map = ewoc_map[ewoc_map["ewoc_code"].notna()]
        ewoc_map["ewoc_code"] = ewoc_map["ewoc_code"].str.replace("-", "").astype(int)
        ewoc_map = ewoc_map.apply(lambda x: x[: x.last_valid_index()].ffill(), axis=1)
        ewoc_map.set_index("ewoc_code", inplace=True)

        df.loc[df["CROPTYPE_LABEL"] == 0, "CROPTYPE_LABEL"] = np.nan
        df["CROPTYPE_LABEL"] = df["CROPTYPE_LABEL"].fillna(df["LANDCOVER_LABEL"])

        df["ewoc_code"] = df["CROPTYPE_LABEL"].map(wc2ewoc_map.set_index("croptype")["ewoc_code"])
        df["landcover_name"] = df["ewoc_code"].map(ewoc_map["landcover_name"])
        df["cropgroup_name"] = df["ewoc_code"].map(ewoc_map["cropgroup_name"])
        df["croptype_name"] = df["ewoc_code"].map(ewoc_map["croptype_name"])

        df["downstream_class"] = df["ewoc_code"].map(
            {int(k): v for k, v in CLASS_MAPPINGS[downstream_classes].items()}
        )
        df["finetune_class"] = df["ewoc_code"].map(
            {int(k): v for k, v in CLASS_MAPPINGS[finetune_classes].items()}
        )
        df["balancing_class"] = df["ewoc_code"].map(
            {int(k): v for k, v in CLASS_MAPPINGS["CROPTYPE19"].items()}
        )

        return df

    @staticmethod
    def check(array: np.ndarray) -> np.ndarray:
        assert not np.isnan(array).any()
        return array

    @staticmethod
    def join_with_world_df(dataframe: pd.DataFrame) -> pd.DataFrame:
        world_df = load_world_df()
        gdataframe = gpd.GeoDataFrame(
            data=dataframe,
            geometry=gpd.GeoSeries.from_xy(x=dataframe.lon, y=dataframe.lat),
            crs="EPSG:4326",
        )
        # project to non geographic CRS, otherwise geopandas gives a warning
        joined = gpd.sjoin_nearest(
            gdataframe.to_crs("EPSG:3857"), world_df.to_crs("EPSG:3857"), how="left"
        )
        joined = joined[~joined.index.duplicated(keep="first")]
        if joined.isna().any(axis=1).any():
            logger.warning("Some coordinates couldn't be matched to a country")
        return joined.to_crs("EPSG:4326")

    @classmethod
    def split_df(
        cls,
        df: pd.DataFrame,
        val_sample_ids: Optional[List[str]] = None,
        val_countries_iso3: Optional[List[str]] = None,
        val_years: Optional[List[int]] = None,
        val_size: Optional[float] = None,
        train_only_samples: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if val_size is not None:
            assert (
                (val_countries_iso3 is None) and (val_years is None) and (val_sample_ids is None)
            )
            val, train = np.split(
                df.sample(frac=1, random_state=DEFAULT_SEED), [int(val_size * len(df))]
            )
            logger.info(f"Using {len(train)} train and {len(val)} val samples")
            return pd.DataFrame(train), pd.DataFrame(val)
        if val_sample_ids is not None:
            assert (val_countries_iso3 is None) and (val_years is None)
            is_val = df.sample_id.isin(val_sample_ids)
            is_train = ~df.sample_id.isin(val_sample_ids)
        elif val_countries_iso3 is not None:
            assert (val_sample_ids is None) and (val_years is None)
            df = cls.join_with_world_df(df)
            for country in val_countries_iso3:
                assert df.iso3.str.contains(
                    country
                ).any(), f"Tried removing {country} but it is not in the dataframe"
            if train_only_samples is not None:
                is_val = df.iso3.isin(val_countries_iso3) & ~df.sample_id.isin(train_only_samples)
            else:
                is_val = df.iso3.isin(val_countries_iso3)
            is_train = ~df.iso3.isin(val_countries_iso3)
        elif val_years is not None:
            df["end_date_ts"] = pd.to_datetime(df.end_date)
            if train_only_samples is not None:
                is_val = df.end_date_ts.dt.year.isin(val_years) & ~df.sample_id.isin(
                    train_only_samples
                )
            else:
                is_val = df.end_date_ts.dt.year.isin(val_years)
            is_train = ~df.end_date_ts.dt.year.isin(val_years)

        logger.info(f"Using {len(is_val) - sum(is_val)} train and {sum(is_val)} val samples")

        return df[is_train], df[is_val]


class WorldCerealMaskedDataset(WorldCerealBase):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        mask_params: MaskParamsNoDw,
        task_type: str = "cropland",
        croptype_list: List = [],
    ):
        super().__init__(dataframe)
        self.mask_params = mask_params
        self.task_type = task_type
        self.croptype_list = croptype_list

    def __getitem__(self, idx):
        # Get the sample
        row = self.df.iloc[idx, :]
        eo, real_mask_per_token, latlon, month, valid_month = self.row_to_arrays(
            row, self.task_type, self.croptype_list
        )
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


def filter_remove_noncrops(df: pd.DataFrame) -> pd.DataFrame:
    labels_to_exclude = [
        0,
        991,
        7900,
        9900,
        9998,  # unspecified cropland
        1910,
        1900,
        1920,
        1000,  # cereals, too generic
        11,
        9910,
        6212,  # temporary crops, too generic
        7920,
        9520,
        3400,
        3900,  # generic and other classes
        4390,
        4000,
        4300,  # generic and other classes
    ]
    df = df[(df["LANDCOVER_LABEL"] == 11) & (~df["CROPTYPE_LABEL"].isin(labels_to_exclude))]
    df.reset_index(inplace=True)
    return df


class WorldCerealLabelledDataset(WorldCerealBase):
    # 0: no information, 10: could be both annual or perennial
    FILTER_LABELS = [0, 10]

    def __init__(
        self,
        dataframe: pd.DataFrame,
        countries_to_remove: Optional[List[str]] = None,
        years_to_remove: Optional[List[int]] = None,
        balance: bool = False,
        task_type: str = "cropland",
        croptype_list: List = [],
        return_hierarchical_labels: bool = False,
        mask_ratio: float = 0.0,
    ):
        dataframe = dataframe.loc[~dataframe.LANDCOVER_LABEL.isin(self.FILTER_LABELS)]

        if countries_to_remove is not None:
            dataframe = self.join_with_world_df(dataframe)
            for country in countries_to_remove:
                assert dataframe.name.str.contains(
                    country
                ).any(), f"Tried removing {country} but it is not in the dataframe"
            dataframe = dataframe[(~dataframe.name.isin(countries_to_remove))]
        if years_to_remove is not None:
            dataframe["end_date"] = pd.to_datetime(dataframe.end_date)
            dataframe = dataframe[(~dataframe.end_date.dt.year.isin(years_to_remove))]

        self._class_weights: Optional[np.ndarray] = None
        self.task_type = task_type
        self.croptype_list = croptype_list
        self.return_hierarchical_labels = return_hierarchical_labels
        self.mask_ratio = mask_ratio
        self.mask_params = MaskParamsNoDw(
            (
                "group_bands",
                "random_timesteps",
                "chunk_timesteps",
                "random_combinations",
            ),
            mask_ratio,
        )

        super().__init__(dataframe)
        if balance:
            if self.task_type == "cropland":
                neg_indices, pos_indices = [], []
                for loc_idx, (_, row) in enumerate(self.df.iterrows()):
                    target = self.target_crop(
                        row, self.task_type, self.croptype_list, return_hierarchical_labels
                    )
                    if target == 0:
                        neg_indices.append(loc_idx)
                    else:
                        pos_indices.append(loc_idx)
                if len(pos_indices) > len(neg_indices):
                    self.indices = (
                        pos_indices + (len(pos_indices) // len(neg_indices)) * neg_indices
                    )
                elif len(neg_indices) > len(pos_indices):
                    self.indices = (
                        neg_indices + (len(neg_indices) // len(pos_indices)) * pos_indices
                    )
                else:
                    self.indices = neg_indices + pos_indices
            if self.task_type == "croptype":
                classes_lst = self.df["balancing_class"].unique()

                # optimal_class_size = self.df["balancing_class"].value_counts().max()
                optimal_class_size = 100
                balancing_coeff = 1.5

                balanced_inds = []
                for tclass in classes_lst:
                    tclass_sample_ids = self.df[self.df["balancing_class"] == tclass].index
                    tclass_loc_idx = [self.df.index.get_loc(xx) for xx in tclass_sample_ids]
                    if len(tclass_loc_idx) < optimal_class_size:

                        if balancing_coeff > 0:
                            if (optimal_class_size / len(tclass_loc_idx)) > balancing_coeff:
                                samples_to_add = np.random.choice(
                                    tclass_loc_idx, size=int(len(tclass_loc_idx) / balancing_coeff)
                                )
                                tclass_loc_idx.extend(list(samples_to_add))
                            else:
                                tclass_loc_idx = tclass_loc_idx * (
                                    optimal_class_size // len(tclass_loc_idx)
                                )
                        else:
                            tclass_loc_idx = tclass_loc_idx * (
                                optimal_class_size // len(tclass_loc_idx)
                            )

                    balanced_inds.extend(tclass_loc_idx)
                self.indices = balanced_inds
        else:
            self.indices = [i for i in range(len(self.df))]

    @staticmethod
    def target_crop(
        row_d: pd.Series,
        task_type: str = "cropland",
        croptype_list: List = [],
        return_hierarchical_labels: bool = False,
    ) -> Union[int, np.ndarray, List]:

        _target: Union[int, np.ndarray, List]
        if task_type == "cropland":
            _target = int(row_d["LANDCOVER_LABEL"] == 11)
        if task_type == "croptype":
            if return_hierarchical_labels:
                _target = [row_d["landcover_name"], row_d["downstream_class"]]
            elif len(croptype_list) == 0:
                _target = row_d["downstream_class"]
            else:
                _target = np.array(row_d[croptype_list].astype(int).values)
        return _target

    @staticmethod
    def multiply_list_length_by_float(input_list: List, multiplier: float) -> List:
        decimal_part, integer_part = modf(multiplier)
        sublist = sample(input_list, k=int(len(input_list) * decimal_part))
        return input_list * int(integer_part) + sublist

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the sample
        df_index = self.indices[idx]
        row = self.df.iloc[df_index, :]
        eo, mask_per_token, latlon, month, valid_month = self.row_to_arrays(
            row, self.task_type, self.croptype_list
        )
        if self.mask_ratio > 0:
            mask_per_token, eo, _, _ = self.mask_params.mask_data(eo, mask_per_token)
        mask_per_variable = np.repeat(mask_per_token, BAND_EXPANSION, axis=1)

        target = self.target_crop(
            row, self.task_type, self.croptype_list, self.return_hierarchical_labels
        )

        return (
            self.normalize_and_mask(eo),
            target,
            np.ones(self.NUM_TIMESTEPS) * (DynamicWorld2020_2021.class_amount),
            latlon,
            month,
            valid_month,
            mask_per_variable,
        )

    @property
    def class_weights(self) -> np.ndarray:
        from sklearn.utils.class_weight import compute_class_weight

        if self._class_weights is None:
            ys: Union[List, np.ndarray]
            ys = []
            for _, row in self.df.iterrows():
                ys.append(
                    self.target_crop(
                        row, self.task_type, self.croptype_list, self.return_hierarchical_labels
                    )
                )
            self._class_weights = compute_class_weight(
                class_weight="balanced", classes=np.unique(ys), y=ys
            )
        return self._class_weights


class WorldCerealLabelled10DDataset(WorldCerealLabelledDataset):

    NUM_TIMESTEPS = 36

    @classmethod
    def get_month_array(cls, row: pd.Series) -> np.ndarray:
        start_date, end_date = datetime.strptime(row.start_date, "%Y-%m-%d"), datetime.strptime(
            row.end_date, "%Y-%m-%d"
        )

        # Calculate the step size for 10-day intervals and create a list of dates
        step = int((end_date - start_date).days / (cls.NUM_TIMESTEPS - 1))
        date_vector = [start_date + timedelta(days=i * step) for i in range(cls.NUM_TIMESTEPS)]

        # Ensure last date is not beyond the end date
        if date_vector[-1] > end_date:
            date_vector[-1] = end_date

        return np.array([d.month - 1 for d in date_vector])

    def __getitem__(self, idx):
        # Get the sample
        df_index = self.indices[idx]
        row = self.df.iloc[df_index, :]
        eo, mask_per_token, latlon, _, valid_month = self.row_to_arrays(
            row, self.task_type, self.croptype_list
        )
        target = self.target_crop(
            row, self.task_type, self.croptype_list, self.return_hierarchical_labels
        )
        if self.mask_ratio > 0:
            mask_per_token, eo, _, _ = self.mask_params.mask_data(eo, mask_per_token)
        mask_per_variable = np.repeat(mask_per_token, BAND_EXPANSION, axis=1)
        return (
            self.normalize_and_mask(eo),
            target,
            np.ones(self.NUM_TIMESTEPS) * (DynamicWorld2020_2021.class_amount),
            latlon,
            self.get_month_array(row),
            valid_month,
            mask_per_variable,
        )


class WorldCerealInferenceDataset(Dataset):
    _NODATAVALUE = 65535
    Y = "WORLDCEREAL_TEMPORARYCROPS_2021"

    def __init__(self, path_to_files: Path = data_dir / "inference_areas"):
        self.path_to_files = path_to_files
        self.all_files = list(self.path_to_files.glob("*.nc"))

    def __len__(self):
        return len(self.all_files)

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

        # Handle NaN values in Presto compatible way
        inarr = inarr.astype(np.float32)
        inarr = inarr.fillna(65535)

        eo_data = np.zeros((num_pixels, num_timesteps, len(BANDS)))
        mask = np.zeros((num_pixels, num_timesteps, len(BANDS_GROUPS_IDX)))

        for presto_band in NORMED_BANDS:
            if presto_band in inarr.coords["bands"]:
                values = np.swapaxes(
                    inarr.sel(bands=presto_band).values.reshape((num_timesteps, -1)),
                    0,
                    1,
                )
                idx_valid = values != cls._NODATAVALUE
                values = cls._preprocess_band_values(values, presto_band)
                eo_data[:, :, BANDS.index(presto_band)] = values * idx_valid
                mask[:, :, IDX_TO_BAND_GROUPS[presto_band]] += ~idx_valid
            elif presto_band == "NDVI":
                # # NDVI will be computed by the normalize function
                continue
            else:
                logger.warning(f"Band {presto_band} not found in input data.")
                eo_data[:, :, BANDS.index(presto_band)] = 0
                mask[:, :, IDX_TO_BAND_GROUPS[presto_band]] = 1

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
        transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        x, y = np.meshgrid(inarr.x, inarr.y)
        lon, lat = transformer.transform(x, y)

        flat_latlons = rearrange(np.stack([lat, lon]), "c x y -> (x y) c")

        # 2D array where each row represents a pair of latitude and longitude coordinates.
        return flat_latlons

    @classmethod
    def _preprocess_band_values(cls, values: np.ndarray, presto_band: str) -> np.ndarray:
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

        start_month = (inarr.t.values[0].astype("datetime64[M]").astype(int) % 12 + 1) - 1

        months = np.ones((num_instances)) * start_month

        return months

    @staticmethod
    def _subset_array_temporally(inarr: xr.DataArray) -> xr.DataArray:
        """
        Subset the input xarray.DataArray temporally based on `valid_time` attribute.

        Args:
            inarr (xr.DataArray): Input xarray.DataArray containing EO data.

        Returns:
            xr.DataArray: Temporally subsetted xarray.DataArray.
        """

        # Use valid_time attribute to extract the right part of the time series
        valid_time = pd.to_datetime(inarr.attrs["valid_time"]).replace(day=1)
        end_time = valid_time + pd.DateOffset(months=5)
        start_time = valid_time - pd.DateOffset(months=6)
        inarr = inarr.sel(t=slice(start_time, end_time))
        num_timesteps = len(inarr.t)
        assert num_timesteps == 12, "Expected 12 timesteps, only found {}".format(num_timesteps)

        return inarr

    @classmethod
    def nc_to_arrays(
        cls, filepath: Path
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        ds = xr.open_dataset(filepath)
        epsg = CRS.from_wkt(xr.open_dataset(filepath).crs.attrs["crs_wkt"]).to_epsg()

        if epsg is None:
            raise ValueError("EPSG code not found in the input file.")
        inarr = ds.drop_vars("crs").to_array(dim="bands")

        # Extract coordinates for reconstruction
        x_coord, y_coord = inarr.x, inarr.y

        # Temporal subsetting to 12 timesteps
        inarr = cls._subset_array_temporally(inarr)

        eo_data, mask = cls._extract_eo_data(inarr)
        flat_latlons = cls._extract_latlons(inarr, epsg)
        months = cls._extract_months(inarr)

        # extract valid_date from ds properties and multiply the was as months array
        # add a test to it if there's none
        valid_month = pd.to_datetime(ds.attrs["valid_time"]).month - 1
        valid_months = np.full_like(months, valid_month)

        if cls.Y not in ds:
            target = np.ones_like(months) * cls._NODATAVALUE
        else:
            target = rearrange(inarr.sel(bands=cls.Y).values, "t x y -> (x y) t")

        return (
            eo_data,
            np.repeat(mask, BAND_EXPANSION, axis=-1),
            flat_latlons,
            months,
            target,
            valid_months,
            x_coord,
            y_coord,
        )

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        eo, mask, flat_latlons, months, target, valid_months, x_coord, y_coord = self.nc_to_arrays(
            filepath
        )

        dynamic_world = np.ones((eo.shape[0], eo.shape[1])) * (DynamicWorld2020_2021.class_amount)

        return (
            S1_S2_ERA5_SRTM.normalize(eo),
            dynamic_world,
            mask,
            flat_latlons,
            months,
            target,
            valid_months,
            x_coord,
            y_coord,
        )

    @staticmethod
    def combine_predictions(
        x_coord: Union[xr.DataArray, np.ndarray, List[float]],
        y_coord: Union[xr.DataArray, np.ndarray, List[float]],
        all_preds: np.ndarray,
        all_preds_ewoc_code: np.ndarray,
        all_probs: np.ndarray,
        gt: np.ndarray,
        ndvi: np.ndarray,
        b2: np.ndarray,
        b3: np.ndarray,
        b4: np.ndarray,
    ) -> xr.DataArray:

        # Get band names
        bands = [
            "ground_truth",
            "ndvi",
            "b2",
            "b3",
            "b4",
            "prob_0",
            "prob_1",
            "prediction_0",
            "pred0_ewoc",
        ]

        if (gt.ndim == 2) and (gt.shape[-1] > 1):
            gt = gt[:, 0]

        x_coord = np.unique(x_coord)
        y_coord = np.unique(y_coord)

        if len(all_probs.shape) == 1:
            all_probs = np.expand_dims(all_probs, axis=-1)

        top1_prob = np.max(all_probs, axis=-1)
        if all_probs.shape[-1] > 1:
            top2_prob = np.partition(all_probs, -2, axis=-1)[:, -2]
        else:
            top2_prob = np.zeros_like(all_preds_ewoc_code)

        # Initialize gridded data array
        data = np.empty((len(bands), len(y_coord), len(x_coord)))

        # Fill with ground truth, NDVI and rgb bands
        data[0, ...] = rearrange(gt, "(y x) -> 1 y x", y=len(y_coord), x=len(x_coord))
        data[1, ...] = rearrange(ndvi, "(y x) -> 1 y x", y=len(y_coord), x=len(x_coord))
        data[2, ...] = rearrange(b2, "(y x) -> 1 y x", y=len(y_coord), x=len(x_coord))
        data[3, ...] = rearrange(b3, "(y x) -> 1 y x", y=len(y_coord), x=len(x_coord))
        data[4, ...] = rearrange(b4, "(y x) -> 1 y x", y=len(y_coord), x=len(x_coord))

        # Fill with gridded probabilities
        data[5, ...] = rearrange(top1_prob, "(y x) -> 1 y x", y=len(y_coord), x=len(x_coord))
        data[6, ...] = rearrange(top2_prob, "(y x) -> 1 y x", y=len(y_coord), x=len(x_coord))

        # Fill with gridded predictions
        data[7, ...] = rearrange(all_preds, "(y x) -> 1 y x", y=len(y_coord), x=len(x_coord))
        data[8, ...] = rearrange(
            all_preds_ewoc_code, "(y x) -> 1 y x", y=len(y_coord), x=len(x_coord)
        )

        return xr.DataArray(coords=[bands, y_coord, x_coord], dims=["bands", "y", "x"], data=data)
