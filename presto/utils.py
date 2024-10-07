import json
import logging
import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import xarray as xr

from .dataops import (
    BANDS,
    ERA5_BANDS,
    MIN_EDGE_BUFFER,
    NODATAVALUE,
    NORMED_BANDS,
    REMOVED_BANDS,
    S1_BANDS,
    S1_S2_ERA5_SRTM,
    S2_BANDS,
    SRTM_BANDS,
    DynamicWorld2020_2021,
)

# plt = None

logger = logging.getLogger("__main__")

data_dir = Path(__file__).parent.parent / "data"
config_dir = Path(__file__).parent.parent / "config"
default_model_path = data_dir / "default_model.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEFAULT_SEED: int = 42


# From https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
def seed_everything(seed: int = DEFAULT_SEED):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def initialize_logging(output_dir: Union[str, Path], to_file=True, logger_name="__main__"):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.setLevel(logging.INFO)

    if to_file:
        path = os.path.join(output_dir, "console-output.log")
        fh = logging.FileHandler(path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("Initialized logging to %s" % path)
    return logger


def timestamp_dirname(suffix: Optional[str] = None) -> str:
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    return f"{ts}_{suffix}" if suffix is not None else ts


def get_class_mappings() -> Dict:
    """Method to get the WorldCereal class mappings for downstream task.

    Returns
    -------
    Dict
        the resulting dictionary with the class mappings
    """
    with open(data_dir / "croptype_mappings" / "croptype_classes.json") as f:
        CLASS_MAPPINGS = json.load(f)

    return CLASS_MAPPINGS


def process_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes in a DataFrame with S1, S2 and ERA5 observations and their respective dates
    in long format and returns it in wide format.

    Each row of the input DataFrame should represent a unique combination
    of sample_id and timestamp, also containing start_date and valid_date columns.
    start_date is the first date of the timeseries.
    valid_date is the date for which the crop of the sample is valid
    (prefrerably it is located around
    the center of the agricultural season, but not necessarily).
    timestamp is the date of the observation.

    This function performs the following operations:
    - computing relative position of the timestamp (timestamp_ind variable)
      and valid_date (valid_position variable) in the timeseries;
    - filtering out samples were valid date is outside the range of the actual extractions
    - adding dummy timesteps filled with NODATA values before the start_date or after
      the end_date for samples where valid_date is close to the edge of the timeseries;
      this closeness is defined by the globally defined parameter MIN_EDGE_BUFFER
    - reinitializing the start_date and timestamp_ind to take into account
      newly added timesteps
    - checking for missing timesteps in the middle of the timeseries and adding them
      with NODATA values
    - pivoting the DataFrame to wide format with columns for each band
      and timesteps as suffixes
    - assigning the correct suffixes to the band names
    - post-processing with prep_dataframe function

    Returns
    -------
    pd.DataFrame
        pivoted DataFrame with columns for each band and timesteps as suffixes

    Raises
    ------
    ValueError
        error is raised if pivot results in an empty DataFrame
    """

    # add dummy value + rename stuff for compatibility with existing functions
    df["OPTICAL-B8A"] = NODATAVALUE

    # TODO: this needs to go away once the transition to new data is complete
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

    # should these definitions be here? or better in the dataops.py?
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
        "location_id",
        "ref_id",
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

    df["timestamp_ind"] = (df["timestamp"].dt.year * 12 + df["timestamp"].dt.month) - (
        df["start_date"].dt.year * 12 + df["start_date"].dt.month
    )
    df["valid_position"] = (df["valid_date"].dt.year * 12 + df["valid_date"].dt.month) - (
        df["start_date"].dt.year * 12 + df["start_date"].dt.month
    )
    df["valid_position_diff"] = df["timestamp_ind"] - df["valid_position"]

    # save the initial start_date for later
    df["initial_start_date"] = df["start_date"].copy()
    index_columns.append("initial_start_date")

    # define samples where valid_date is outside the range of the actual extractions
    # and remove them from the dataset
    latest_obs_position = df.groupby(["sample_id"])[
        ["valid_position", "timestamp_ind", "valid_position_diff"]
    ].max()
    df["is_last_available_ts"] = (
        df["sample_id"].map(latest_obs_position["timestamp_ind"]) == df["timestamp_ind"]
    )
    samples_after_end_date = latest_obs_position[
        (latest_obs_position["valid_position"] > latest_obs_position["timestamp_ind"])
    ].index
    samples_before_start_date = latest_obs_position[
        (latest_obs_position["valid_position"] < 0)
    ].index

    if len(samples_after_end_date) > 0 or len(samples_before_start_date) > 0:
        logger.warning(
            f"""\
Dataset {df["ref_id"].iloc[0]}: removing {len(samples_after_end_date)}\
samples with valid_date after the end_date\
and {len(samples_before_start_date)} samples with valid_date before the start_date"""
        )
        df = df[~df["sample_id"].isin(samples_before_start_date)]
        df = df[~df["sample_id"].isin(samples_after_end_date)]

    # add timesteps before the start_date where needed
    for n_ts_to_add in range(1, MIN_EDGE_BUFFER + 1):
        samples_to_add_ts_before_start = latest_obs_position[
            (MIN_EDGE_BUFFER - latest_obs_position["valid_position"]) >= -n_ts_to_add
        ].index
        dummy_df = df[
            (df["sample_id"].isin(samples_to_add_ts_before_start)) & (df["timestamp_ind"] == 0)
        ].copy()
        dummy_df["timestamp"] = dummy_df["timestamp"] - pd.DateOffset(
            months=n_ts_to_add
        )  # type: ignore
        dummy_df[feature_columns] = NODATAVALUE
        df = pd.concat([df, dummy_df])

    # add timesteps after the end_date where needed
    for n_ts_to_add in range(1, MIN_EDGE_BUFFER + 1):
        samples_to_add_ts_after_end = latest_obs_position[
            (MIN_EDGE_BUFFER - latest_obs_position["valid_position_diff"]) >= n_ts_to_add
        ].index
        dummy_df = df[
            (df["sample_id"].isin(samples_to_add_ts_after_end)) & (df["is_last_available_ts"])
        ].copy()
        dummy_df["timestamp"] = dummy_df["timestamp"] + pd.DateOffset(
            months=n_ts_to_add
        )  # type: ignore
        dummy_df[feature_columns] = NODATAVALUE
        df = pd.concat([df, dummy_df])

    # Now reassign start_date to the minimum timestamp
    new_start_date = df.groupby(["sample_id"])["timestamp"].min()
    df["start_date"] = df["sample_id"].map(new_start_date)

    # reinitialize timestep_ind
    df["timestamp_ind"] = (df["timestamp"].dt.year * 12 + df["timestamp"].dt.month) - (
        df["start_date"].dt.year * 12 + df["start_date"].dt.month
    )

    # check for missing timestamps in the middle of timeseries
    # and create corresponding columns with NODATAVALUE
    missing_timestamps = [
        xx for xx in range(df["timestamp_ind"].max()) if xx not in df["timestamp_ind"].unique()
    ]
    present_timestamps = [
        xx for xx in range(df["timestamp_ind"].max()) if xx not in missing_timestamps
    ]
    for missing_timestamp in missing_timestamps:
        dummy_df = df[df["timestamp_ind"] == np.random.choice(present_timestamps)].copy()
        dummy_df["timestamp_ind"] = missing_timestamp
        dummy_df[feature_columns] = NODATAVALUE
        df = pd.concat([df, dummy_df])

    # finally pivot the dataframe
    df_pivot = df.pivot(index=index_columns, columns="timestamp_ind", values=feature_columns)
    df_pivot = df_pivot.fillna(NODATAVALUE)

    if df_pivot.empty:
        raise ValueError("Left with an empty DataFrame!")

    df_pivot.reset_index(inplace=True)
    df_pivot.columns = [
        f"{xx[0]}-ts{xx[1]}" if isinstance(xx[1], int) else xx[0]
        for xx in df_pivot.columns.to_flat_index()
    ]  # type: ignore
    df_pivot.columns = [
        f"{xx}-10m" if any(band in xx for band in bands10m) else xx for xx in df_pivot.columns
    ]  # type: ignore
    df_pivot.columns = [
        f"{xx}-20m" if any(band in xx for band in bands20m) else xx for xx in df_pivot.columns
    ]  # type: ignore
    df_pivot.columns = [
        f"{xx}-100m" if any(band in xx for band in bands100m) else xx for xx in df_pivot.columns
    ]  # type: ignore

    df_pivot["valid_position"] = (
        df_pivot["valid_date"].dt.year * 12 + df_pivot["valid_date"].dt.month
    ) - (df_pivot["start_date"].dt.year * 12 + df_pivot["start_date"].dt.month)
    df_pivot["available_timesteps"] = (
        df_pivot["end_date"].dt.year * 12 + df_pivot["end_date"].dt.month
    ) - (df_pivot["start_date"].dt.year * 12 + df_pivot["start_date"].dt.month)

    df_pivot["year"] = df_pivot["valid_date"].dt.year

    df_pivot["start_date"] = df_pivot["start_date"].dt.date.astype(str)
    df_pivot["end_date"] = df_pivot["end_date"].dt.date.astype(str)
    df_pivot["valid_date"] = df_pivot["valid_date"].dt.date.astype(str)

    df_pivot = prep_dataframe(df_pivot)

    return df_pivot


def construct_single_presto_input(
    s1: Optional[torch.Tensor] = None,
    s1_bands: Optional[List[str]] = None,
    s2: Optional[torch.Tensor] = None,
    s2_bands: Optional[List[str]] = None,
    era5: Optional[torch.Tensor] = None,
    era5_bands: Optional[List[str]] = None,
    srtm: Optional[torch.Tensor] = None,
    srtm_bands: Optional[List[str]] = None,
    dynamic_world: Optional[torch.Tensor] = None,
    normalize: bool = True,
):
    """
    Inputs are paired into a tensor input <X> and a list <X>_bands, which describes <X>.

    <X> should have shape (num_timesteps, len(<X>_bands)), with the following bands possible for
    each input:

    s1: ["VV", "VH"]
    s2: ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
    era5: ["temperature_2m", "total_precipitation"]
        "temperature_2m": Temperature of air at 2m above the surface of land,
            sea or in-land waters in Kelvin (K)
        "total_precipitation": Accumulated liquid and frozen water, including rain and snow,
            that falls to the Earth's surface. Measured in metres (m)
    srtm: ["elevation", "slope"]

    dynamic_world is a 1d input of shape (num_timesteps,) representing the dynamic world classes
        of each timestep for that pixel
    """
    num_timesteps_list = [x.shape[0] for x in [s1, s2, era5, srtm] if x is not None]
    if dynamic_world is not None:
        num_timesteps_list.append(len(dynamic_world))

    assert len(num_timesteps_list) > 0
    assert all(num_timesteps_list[0] == timestep for timestep in num_timesteps_list)
    num_timesteps = num_timesteps_list[0]
    mask, x = torch.ones(num_timesteps, len(BANDS)), torch.zeros(num_timesteps, len(BANDS))

    for band_group in [
        (s1, s1_bands, S1_BANDS),
        (s2, s2_bands, S2_BANDS),
        (era5, era5_bands, ERA5_BANDS),
        (srtm, srtm_bands, SRTM_BANDS),
    ]:
        data, input_bands, output_bands = band_group
        if data is not None:
            assert input_bands is not None
        else:
            continue

        kept_output_bands = [x for x in output_bands if x not in REMOVED_BANDS]
        # construct a mapping from the input bands to the expected bands
        kept_input_band_idxs = [i for i, val in enumerate(input_bands) if val in kept_output_bands]
        kept_input_band_names = [val for val in input_bands if val in kept_output_bands]

        input_to_output_mapping = [BANDS.index(val) for val in kept_input_band_names]

        x[:, input_to_output_mapping] = data[:, kept_input_band_idxs]
        mask[:, input_to_output_mapping] = 0

    if dynamic_world is None:
        dynamic_world = torch.ones(num_timesteps) * (DynamicWorld2020_2021.class_amount)

    keep_indices = [idx for idx, val in enumerate(BANDS) if val != "B9"]
    mask = mask[:, keep_indices]

    if normalize:
        # normalize includes x = x[:, keep_indices]
        x = S1_S2_ERA5_SRTM.normalize(x)
        if s2_bands is not None:
            if ("B8" in s2_bands) and ("B4" in s2_bands):
                mask[:, NORMED_BANDS.index("NDVI")] = 0
    else:
        x = x[:, keep_indices]
    return x, mask, dynamic_world


def plot_results(
    world_df: gpd.GeoDataFrame,
    metrics: Dict,
    output_dir: Path,
    epoch: Optional[int] = None,
    show: bool = False,
    to_wandb: bool = False,
    prefix: str = "",
):

    from matplotlib import pyplot as plt

    def plot(title: str, plot_fn: Callable, figsize=(15, 5)) -> Path:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot_fn(ax=ax)
        if epoch is not None:
            title = f"{prefix}{title} - epoch {epoch}"
        plt.title(title)
        plt.tight_layout()
        path = output_dir / f"{title.replace(' ', '_')}.png"
        plt.savefig(path)
        if show:
            plt.show()
        plt.close()
        return path

    def plot_map(scores: gpd.GeoDataFrame, ax, vmin=0, vmax=1, cmap="coolwarm"):
        scores.plot(column="value", legend=True, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap)

    def plot_year(scores: pd.DataFrame, ax, ymin=0, ymax=1, ylabel=""):
        scores.loc[:, ["year", "value"]].plot(kind="bar", legend=False, ax=ax)
        plt.xticks(ticks=range(len(scores.index)), labels=scores.year)
        plt.ylim(ymin, ymax)
        plt.ylabel(ylabel)

    def plot_for_group(grp_df):
        mrgd_country = world_df.merge(grp_df, left_on="name", right_on="country", how="left")
        mrgd_country = mrgd_country.dropna(subset="model")

        grp_df_aez = grp_df.loc[~pd.isna(grp_df.aez)]
        grp_df_aez.loc[:, "aez"] = grp_df_aez.aez.astype(int)
        mrgd_aez = aez_df.merge(grp_df_aez, left_on="zoneID", right_on="aez", how="left")
        mrgd_aez = mrgd_aez.dropna(subset="model")

        grp_df_y = grp_df.loc[pd.notna(grp_df.year)].sort_values("year")
        grp_df_y.loc[:, "year"] = grp_df_y.year.astype(str)

        name = " ".join(grp_df.name)
        model, metric_type = grp_df.name
        not_1_as_max = ("positives", "predicted", "samples")
        upper_cntry = mrgd_country.value.max() if metric_type in not_1_as_max else 1
        upper_aez = mrgd_aez.value.max() if metric_type in not_1_as_max else 1
        upper_y = (1.1 * grp_df_y.value.max()) if metric_type in not_1_as_max else 1
        img_paths = [
            plot(f"{name} Country", partial(plot_map, mrgd_country, vmax=upper_cntry)),
            plot(f"{name} AEZ", partial(plot_map, mrgd_aez, vmax=upper_aez)),
            plot(
                f"{name} Year",
                partial(plot_year, grp_df_y, ymax=upper_y, ylabel=metric_type),
                (6, 5),
            ),
        ]

        if model != "CatBoost" and metric_type in ("f1", "precision", "recall"):
            diff_country = mrgd_country.copy()
            diff_country["value"] -= diff_country["value_catboost"]
            diff_aez = mrgd_aez.copy()
            diff_aez["value"] -= diff_aez["value_catboost"]
            diff_y = grp_df_y.copy()
            diff_y["value"] -= diff_y["value_catboost"]

            img_paths += [
                plot(
                    f"{name} Country - CatBoost",
                    partial(plot_map, diff_country, vmin=-1, cmap="coolwarm"),
                ),
                plot(
                    f"{name} AEZ - CatBoost",
                    partial(plot_map, diff_aez, vmin=-1, cmap="coolwarm"),
                ),
                plot(
                    f"{name} Year - CatBoost",
                    partial(plot_year, diff_y, ymin=-1, ylabel=metric_type),
                    (6, 5),
                ),
            ]

        if to_wandb:
            import wandb

            wandb.log({str(p): wandb.Image(str(p)) for p in img_paths})

    aez_df = gpd.read_file(data_dir / "AEZ.geojson")
    aez_df = aez_df.loc[:, ["zoneID", "geometry"]]
    aez_df.zoneID = aez_df.zoneID.astype(int)

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["value"])
    metrics_df = metrics_df.reset_index(names="metric")
    metrics_df = metrics_df.loc[~metrics_df.metric.str.contains("macro")]

    country = metrics_df.metric.apply(
        lambda m: m.split(":")[-1].lstrip() if "country" in m else None
    )
    aez = metrics_df.metric.apply(lambda m: m.split(":")[-1].lstrip() if "aez" in m else None)
    year = metrics_df.metric.apply(lambda m: m.split(":")[-1].lstrip() if "year" in m else None)
    model = metrics_df.metric.apply(lambda m: m.split("_")[1].strip())
    metrics_df = pd.concat((metrics_df, model, aez, year, country), axis=1)
    metrics_df.columns = pd.Index(["metric", "value", "model", "aez", "year", "country"])

    # e.g. f1, aez_recall: 46172, ...
    metrics_df["metric_wo_model"] = metrics_df.metric.str.split("_").apply(
        lambda x: "_".join(x[-2:]) if x[-2] not in metrics_df.model.unique() else x[-1]
    )
    # e.g. f1, recall, precision
    metrics_df["metric_type"] = metrics_df.metric_wo_model.str.split(":", expand=True).loc[:, 0]
    metrics_df["metric_type"] = metrics_df["metric_type"].str.split("_").apply(lambda x: x[-1])
    # add catboost performance to other model's rows to plot difference
    metrics_df = metrics_df.merge(
        metrics_df.loc[metrics_df.model == "CatBoost"],
        how="left",
        on=["metric_wo_model", "aez", "year", "country"],
        suffixes=(None, "_catboost"),
    )

    metrics_df.groupby(["model", "metric_type"]).apply(plot_for_group)


def plot_spatial(
    spatial_preds: xr.Dataset,
    output_path: Path,
    to_wandb: bool = False,
    task_type: str = "cropland",
):

    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    CLASS_MAPPINGS = get_class_mappings()

    croptype_map = CLASS_MAPPINGS["CROPTYPE0"]
    colors_map = CLASS_MAPPINGS["CROPTYPE0_COLORS"]

    b2 = spatial_preds.sel(bands="b2")["__xarray_dataarray_variable__"].values
    b3 = spatial_preds.sel(bands="b3")["__xarray_dataarray_variable__"].values
    b4 = spatial_preds.sel(bands="b4")["__xarray_dataarray_variable__"].values
    ground_truth = spatial_preds.sel(bands="ground_truth")["__xarray_dataarray_variable__"].values
    ndvi = spatial_preds.sel(bands="ndvi")["__xarray_dataarray_variable__"].values
    pred0_ewoc = spatial_preds.sel(bands="pred0_ewoc")["__xarray_dataarray_variable__"].values
    prediction_0 = spatial_preds.sel(bands="prediction_0")["__xarray_dataarray_variable__"].values
    prob_0 = spatial_preds.sel(bands="prob_0")["__xarray_dataarray_variable__"].values
    prob_1 = spatial_preds.sel(bands="prob_1")["__xarray_dataarray_variable__"].values

    rgb_ts6 = np.dstack(
        (
            (b4 - b4.min()) / (b4.max() - b4.min()),
            (b3 - b3.min()) / (b3.max() - b3.min()),
            (b2 - b2.min()) / (b2.max() - b2.min()),
        )
    )

    fig = plt.figure(figsize=(40, 25))

    fig.add_subplot(2, 3, 1)
    plt.imshow(ground_truth)
    plt.axis("off")
    plt.title("Phase I WorldCereal Mask")

    fig.add_subplot(2, 3, 2)
    plt.imshow(rgb_ts6)
    plt.axis("off")
    plt.title("RGB TS6")

    fig.add_subplot(2, 3, 3)
    plt.imshow(ndvi)
    plt.axis("off")
    plt.title("NDVI TS6")

    if task_type == "croptype":
        fig.add_subplot(2, 3, 4)

        pred0_ewoc_int = [
            int(xx) if not np.isnan(xx) else 1000000000 for xx in np.unique(pred0_ewoc)
        ]
        values = [croptype_map[str(xx)] for xx in pred0_ewoc_int]
        colors = [colors_map[str(xx)] for xx in pred0_ewoc_int]

        # values = [croptype_map[str(xx)] for xx in np.unique(spatial_preds.pred0_ewoc)]
        # colors = [colors_map[str(xx)] for xx in np.unique(spatial_preds.pred0_ewoc)]

        cmap = mcolors.ListedColormap(colors)
        cmap.set_bad(color="whitesmoke")

        plt.imshow(prediction_0, cmap=cmap)
        patches = [mpatches.Patch(color=colors[ii], label=values[ii]) for ii in range(len(values))]
        plt.legend(
            handles=patches,
            bbox_to_anchor=(1.25, 0.65),
            loc=1,
            borderaxespad=0.0,
            prop={"size": 6},
        )
        plt.axis("off")
        plt.title("Croptype predictions")

    if task_type == "cropland":
        fig.add_subplot(2, 3, 4)
        plt.imshow(prob_0 > 0.5)
        plt.axis("off")
        plt.title("Cropland predictions")

    fig.add_subplot(2, 3, 5)
    plt.imshow(prob_0, cmap="Greens", vmin=0, vmax=1)
    plt.colorbar()
    plt.axis("off")
    plt.title("Top1 class prob")

    if task_type == "croptype":
        fig.add_subplot(2, 3, 6)
        plt.imshow(prob_1, cmap="Greens", vmin=0, vmax=1)
        plt.colorbar()
        plt.axis("off")
        plt.title("Top2 class prob")

    # plt.suptitle(test_patch_name)

    plt.savefig(output_path, bbox_inches="tight")
    if to_wandb:
        import wandb

        wandb.log({str(output_path): wandb.Image(str(output_path))})
    plt.close()  # type: ignore


def load_world_df() -> gpd.GeoDataFrame:
    # this could be memoized, but it should only be called 2 or 3 times in a run
    filename = "world-administrative-boundaries/world-administrative-boundaries.shp"
    world_df = gpd.read_file(data_dir / filename)
    world_df = world_df.drop(columns=["status", "color_code", "iso_3166_1_"])
    return world_df


def prep_dataframe(
    df: pd.DataFrame,
    filter_function: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    dekadal: bool = False,
):
    """Duplication from eval.py but otherwise we would need catboost during
    presto inference on OpenEO.
    """
    # SAR cannot equal 0.0 since we take the log of it
    cols = [f"SAR-{s}-ts{t}-20m" for s in ["VV", "VH"] for t in range(36 if dekadal else 12)]

    df = df.drop_duplicates(subset=["sample_id", "lat", "lon", "end_date"])
    df = df[~pd.isna(df).any(axis=1)]
    df = df[~(df.loc[:, cols] == 0.0).any(axis=1)]
    df = df.set_index("sample_id")
    if filter_function is not None:
        df = filter_function(df)
    return df
