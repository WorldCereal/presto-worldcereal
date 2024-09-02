import json
import logging
from datetime import datetime, timedelta
from math import modf
from pathlib import Path
from random import sample
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from einops import rearrange, repeat
from pyproj import Transformer
from rasterio import CRS
from sklearn.utils.class_weight import compute_class_weight
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
        model_mode: str = "",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int, Union[int, str, np.ndarray, List]]:
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

        _target = WorldCerealBase.target_crop(row, task_type, croptype_list, model_mode)

        return (cls.check(eo_data), mask.astype(bool), latlon, month, valid_month, _target)

    def __getitem__(self, idx):
        raise NotImplementedError

    @staticmethod
    def target_crop(
        row_d: pd.Series,
        task_type: str = "cropland",
        croptype_list: List = [],
        model_mode: str = "",
    ) -> Union[int, np.ndarray, List]:

        _target: Union[int, np.ndarray, List]
        if task_type == "cropland":
            _target = int(row_d["LANDCOVER_LABEL"] == 11)
        if task_type == "croptype":
            if model_mode == "Hierarchical CatBoostClassifier":
                _target = [row_d["landcover_name"], row_d["downstream_class"]]
            elif len(croptype_list) == 0:
                _target = row_d["downstream_class"]
            else:
                _target = np.array(row_d[croptype_list].astype(int).values)
        return _target

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
        model_mode: str = "",
    ):
        super().__init__(dataframe)
        self.mask_params = mask_params
        self.task_type = task_type
        self.croptype_list = croptype_list
        self.model_mode = model_mode

    def __getitem__(self, idx):
        # Get the sample
        row = self.df.iloc[idx, :]
        eo, real_mask_per_token, latlon, month, _, _ = self.row_to_arrays(
            row, self.task_type, self.croptype_list, self.model_mode
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
        # target_function: Optional[Callable] = None,
        balance: bool = False,
        task_type: str = "cropland",
        croptype_list: List = [],
        model_mode: str = "",
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
        self.model_mode = model_mode

        super().__init__(dataframe)
        if balance:
            if self.task_type == "cropland":
                neg_indices, pos_indices = [], []
                for loc_idx, (_, row) in enumerate(self.df.iterrows()):
                    target = self.target_crop(
                        row, self.task_type, self.croptype_list, self.model_mode
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
                optimal_class_size = self.df["balancing_class"].value_counts().max()
                balanced_inds = []
                for tclass in classes_lst:
                    tclass_sample_ids = self.df[self.df["balancing_class"] == tclass].index
                    tclass_loc_idx = [self.df.index.get_loc(xx) for xx in tclass_sample_ids]
                    if len(tclass_loc_idx) < optimal_class_size:
                        tclass_loc_idx = tclass_loc_idx * (
                            optimal_class_size // len(tclass_loc_idx)
                        )
                    balanced_inds.extend(tclass_loc_idx)
                self.indices = balanced_inds
        else:
            self.indices = [i for i in range(len(self.df))]

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
        eo, mask_per_token, latlon, month, valid_month, target = self.row_to_arrays(
            row, self.task_type, self.croptype_list, self.model_mode
        )
        mask_per_variable = np.repeat(mask_per_token, BAND_EXPANSION, axis=1)
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
        if self._class_weights is None:
            ys: Union[List, np.ndarray]
            ys = []
            for _, row in self.df.iterrows():
                ys.append(
                    self.target_crop(row, self.task_type, self.croptype_list, self.model_mode)
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
        eo, mask_per_token, latlon, _, valid_month, target = self.row_to_arrays(
            row, self.task_type, self.croptype_list, self.model_mode
        )
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
    Y = "worldcereal_cropland"
    BAND_MAPPING = {
        "B02": "B2",
        "B03": "B3",
        "B04": "B4",
        "B05": "B5",
        "B06": "B6",
        "B07": "B7",
        "B08": "B8",
        # B8A is missing
        "B11": "B11",
        "B12": "B12",
        "VH": "VH",
        "VV": "VV",
        "precipitation-flux": "total_precipitation",
        "temperature-mean": "temperature_2m",
    }

    def __init__(self, path_to_files: Path = data_dir / "inference_areas"):
        self.path_to_files = path_to_files
        self.all_files = list(self.path_to_files.glob("*.nc"))

    def __len__(self):
        return len(self.all_files)

    @classmethod
    def nc_to_arrays(
        cls, filepath: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        ds = xr.open_dataset(filepath)
        epsg_coords = CRS.from_wkt(ds.crs.crs_wkt).to_epsg()

        num_instances = len(ds.x) * len(ds.y)
        num_timesteps = len(ds.t)
        eo_data = np.zeros((num_instances, num_timesteps, len(BANDS)))
        mask = np.zeros((num_instances, num_timesteps, len(BANDS_GROUPS_IDX)))
        # for now, B8A is missing
        mask[:, :, IDX_TO_BAND_GROUPS["B8A"]] = 1

        for org_band, presto_val in cls.BAND_MAPPING.items():
            # flatten the values
            values = np.swapaxes(ds[org_band].values.reshape((num_timesteps, -1)), 0, 1)
            idx_valid = values != cls._NODATAVALUE

            if presto_val in ["VV", "VH"]:
                # convert to dB
                values = 20 * np.log10(values) - 83
            elif presto_val == "total_precipitation":
                # scaling, and AgERA5 is in mm, Presto expects m
                values = values / (100 * 1000.0)
            elif presto_val == "temperature_2m":
                # remove scaling
                values = values / 100

            eo_data[:, :, BANDS.index(presto_val)] = values
            mask[:, :, IDX_TO_BAND_GROUPS[presto_val]] += ~idx_valid

        y = rearrange(ds[cls.Y].values, "t x y -> (x y) t")
        # -1 because we index from 0
        start_month = (ds.t.values[0].astype("datetime64[M]").astype(int) % 12 + 1) - 1
        months = np.ones((num_instances)) * start_month
        valid_month = pd.to_datetime(np.sort(ds.t.values)[6]).month - 1
        valid_months = np.array(num_instances * [valid_month])

        transformer = Transformer.from_crs(f"EPSG:{epsg_coords}", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(ds.x, ds.y)

        latlons = np.stack(
            [np.repeat(lat, repeats=len(lon)), repeat(lon, "c -> (h c)", h=len(lat))],
            axis=-1,
        )

        return eo_data, np.repeat(mask, BAND_EXPANSION, axis=-1), latlons, months, y, valid_months

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        eo, mask, latlons, months, y, valid_months = self.nc_to_arrays(filepath)

        dynamic_world = np.ones((eo.shape[0], eo.shape[1])) * (DynamicWorld2020_2021.class_amount)

        return S1_S2_ERA5_SRTM.normalize(eo), dynamic_world, mask, latlons, months, y, valid_months

    @staticmethod
    def combine_predictions(
        latlons: np.ndarray,
        all_preds: np.ndarray,
        all_preds_ewoc_code: np.ndarray,
        all_probs: np.ndarray,
        gt: np.ndarray,
        ndvi: np.ndarray,
        b2: np.ndarray,
        b3: np.ndarray,
        b4: np.ndarray,
    ) -> pd.DataFrame:

        flat_lat, flat_lon = latlons[:, 0], latlons[:, 1]
        data_dict: Dict[str, np.ndarray] = {"lat": flat_lat, "lon": flat_lon}

        if len(all_probs.shape) == 1:
            all_probs = np.expand_dims(all_probs, axis=-1)

        if len(all_preds.shape) > 1:
            all_preds = all_preds.flatten()

        top1_prob = np.max(all_probs, axis=-1)
        if all_probs.shape[-1] > 1:
            top2_prob = np.partition(all_probs, -2, axis=-1)[:, -2]
        else:
            top2_prob = np.zeros_like(all_preds_ewoc_code)

        data_dict["prob_0"] = top1_prob
        data_dict["prob_1"] = top2_prob
        data_dict["prediction_0"] = all_preds
        data_dict["ground_truth"] = gt[:, 0]
        data_dict["ndvi"] = ndvi
        data_dict["b2"] = b2
        data_dict["b3"] = b3
        data_dict["b4"] = b4
        data_dict["pred0_ewoc"] = all_preds_ewoc_code

        return pd.DataFrame(data=data_dict).set_index(["lat", "lon"])
