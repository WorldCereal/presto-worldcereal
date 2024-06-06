import logging
import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import geopandas as gpd
import pandas as pd
import torch
import xarray as xr

from .dataops import (
    BANDS,
    ERA5_BANDS,
    NORMED_BANDS,
    REMOVED_BANDS,
    S1_BANDS,
    S1_S2_ERA5_SRTM,
    S2_BANDS,
    SRTM_BANDS,
    DynamicWorld2020_2021,
)

plt = None

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
    global plt
    if plt is None:
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
                    f"{name} AEZ - CatBoost", partial(plot_map, diff_aez, vmin=-1, cmap="coolwarm")
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
    metrics_df.columns = ["metric", "value", "model", "aez", "year", "country"]

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


def plot_spatial(spatial_preds: xr.Dataset, output_path: Path, to_wandb: bool = False):
    global plt
    if plt is None:
        from matplotlib import pyplot as plt
    plt.clf()  # type: ignore
    _, axs = plt.subplots(ncols=4, figsize=(25, 4))  # type: ignore
    spatial_preds.ndvi.plot(ax=axs[0], vmin=0, vmax=1)
    spatial_preds.ground_truth.plot(ax=axs[1], vmin=0, vmax=1)
    spatial_preds.prediction_0.plot(ax=axs[2], vmin=0, vmax=1)
    (spatial_preds.prediction_0 > 0.5).plot(ax=axs[3], vmin=0, vmax=1)
    plt.savefig(output_path, bbox_inches="tight")  # type: ignore
    if to_wandb:
        import wandb

        wandb.log({str(output_path): wandb.Image(str(output_path))})
    plt.close()  # type: ignore


def load_world_df() -> pd.DataFrame:
    # this could be memoized, but it should only be called 2 or 3 times in a run
    filename = "world-administrative-boundaries/world-administrative-boundaries.shp"
    world_df = gpd.read_file(data_dir / filename)
    world_df = world_df.drop(columns=["status", "color_code", "iso_3166_1_"])
    return world_df
