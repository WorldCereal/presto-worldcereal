# presto_pretrain_finetune, but in a notebook
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, cast

import pandas as pd
import torch
import xarray as xr

from presto.dataset import WorldCerealBase
from presto.eval import WorldCerealEval
from presto.presto import Presto
from presto.utils import (
    DEFAULT_SEED,
    config_dir,
    data_dir,
    default_model_path,
    device,
    initialize_logging,
    plot_spatial,
    seed_everything,
    timestamp_dirname,
)

logger = logging.getLogger("__main__")

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", type=str, default="")
argparser.add_argument("--path_to_config", type=str, default="")
argparser.add_argument(
    "--output_dir",
    type=str,
    default="",
    help="Parent directory to save output to, <output_dir>/wandb/ "
    "and <output_dir>/output/ will be written to. "
    "Leave empty to use the directory you are running this file from.",
)
argparser.add_argument("--seed", type=int, default=DEFAULT_SEED)
argparser.add_argument("--num_workers", type=int, default=4)
argparser.add_argument("--wandb", dest="wandb", action="store_true")
argparser.add_argument("--wandb_org", type=str, default="nasa-harvest")
argparser.add_argument("--parquet_file", type=str, default="rawts-monthly_calval.parquet")
argparser.add_argument("--val_samples_file", type=str, default="cropland_test_split_samples.csv")
argparser.add_argument("--train_only_samples_file", type=str, default="train_only_samples.csv")
argparser.add_argument("--warm_start", dest="warm_start", action="store_true")
argparser.set_defaults(wandb=False)
argparser.set_defaults(warm_start=True)
args = argparser.parse_args().__dict__

model_name = args["model_name"]
seed: int = args["seed"]
num_workers: int = args["num_workers"]
path_to_config = args["path_to_config"]
warm_start = args["warm_start"]
wandb_enabled: bool = args["wandb"]
wandb_org: str = args["wandb_org"]

seed_everything(seed)
output_parent_dir = Path(args["output_dir"]) if args["output_dir"] else Path(__file__).parent
run_id = None

if wandb_enabled:
    import wandb

    run = wandb.init(
        entity=wandb_org,
        project="presto-worldcereal",
        dir=output_parent_dir,
    )
    run_id = cast(wandb.sdk.wandb_run.Run, run).id

model_logging_dir = output_parent_dir / "output" / timestamp_dirname(run_id)
model_logging_dir.mkdir(exist_ok=True, parents=True)
initialize_logging(model_logging_dir)
logger.info("Using output dir: %s" % model_logging_dir)

parquet_file: str = args["parquet_file"]
val_samples_file: str = args["val_samples_file"]
train_only_samples_file: str = args["train_only_samples_file"]

dekadal = False
if "10d" in parquet_file:
    dekadal = True

path_to_config = config_dir / "default.json"
model_kwargs = json.load(Path(path_to_config).open("r"))

logger.info("Setting up dataloaders")

df = pd.read_parquet(data_dir / parquet_file)

logger.info("Setting up model")
if warm_start:
    model_kwargs = json.load(Path(config_dir / "default.json").open("r"))
    model = Presto.load_pretrained()
    best_model_path: Optional[Path] = default_model_path
else:
    if path_to_config == "":
        path_to_config = config_dir / "default.json"
    model_kwargs = json.load(Path(path_to_config).open("r"))
    model = Presto.construct(**model_kwargs)
    best_model_path = None
model.to(device)

model_modes = ["Random Forest", "Regression", "CatBoostClassifier"]

# 1. Using the provided split
val_samples_df = pd.read_csv(data_dir / val_samples_file)
train_df, test_df = WorldCerealBase.split_df(df, val_sample_ids=val_samples_df.sample_id.tolist())
full_eval = WorldCerealEval(
    train_df, test_df, spatial_inference_savedir=model_logging_dir, dekadal=dekadal
)
results, finetuned_model = full_eval.finetuning_results(model, sklearn_model_modes=model_modes)
logger.info(json.dumps(results, indent=2))

model_path = model_logging_dir / Path("models")
model_path.mkdir(exist_ok=True, parents=True)
finetuned_model_path = model_path / "finetuned_model_stratified.pt"
torch.save(finetuned_model.state_dict(), finetuned_model_path)

train_only_samples = pd.read_csv(data_dir / train_only_samples_file).sample_id.tolist()
# 2. Split according to the countries
country_eval = WorldCerealEval(
    *WorldCerealBase.split_df(
        df,
        val_countries_iso3=["ESP", "NGA", "LVA", "TZA", "ETH", "ARG"],
        train_only_samples=train_only_samples,
    ),
    dekadal=dekadal,
)
country_results, country_finetuned_model = country_eval.finetuning_results(
    model, sklearn_model_modes=model_modes
)
logger.info(json.dumps(country_results, indent=2))

finetuned_model_path_countries = model_path / "finetuned_model_countries.pt"
torch.save(country_finetuned_model.state_dict(), finetuned_model_path_countries)

# 3. Split by year
year_eval = WorldCerealEval(
    *WorldCerealBase.split_df(df, val_years=[2021], train_only_samples=train_only_samples),
    dekadal=dekadal,
)
year_results, year_finetuned_model = year_eval.finetuning_results(
    model, sklearn_model_modes=model_modes
)
logger.info(json.dumps(year_results, indent=2))

all_spatial_preds = list(model_logging_dir.glob("*.nc"))
for spatial_preds_path in all_spatial_preds:
    preds = xr.load_dataset(spatial_preds_path)
    output_path = model_logging_dir / f"{spatial_preds_path.stem}.png"
    plot_spatial(preds, output_path, to_wandb=False)

if wandb_enabled:
    wandb.log(results)
    wandb.log(country_results)
    wandb.log(year_results)

if wandb_enabled and run:
    run.finish()
    logger.info(f"Wandb url: {run.url}")
