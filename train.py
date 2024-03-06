# presto_pretrain_finetune, but in a notebook
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, cast

import pandas as pd
import torch
import xarray as xr

from presto.dataset import WorldCerealMaskedDataset as WorldCerealDataset
from presto.dataset import filter_remove_noncrops, target_maize
from presto.eval import WorldCerealEval, WorldCerealFinetuning
from presto.presto import Presto
from presto.utils import (
    DEFAULT_SEED,
    config_dir,
    data_dir,
    default_model_path,
    device,
    initialize_logging,
    load_world_df,
    plot_results,
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
argparser.add_argument("--val_samples_file", type=str, default="VAL_samples.csv")
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

path_to_config = config_dir / "default.json"
model_kwargs = json.load(Path(path_to_config).open("r"))


df = pd.read_parquet(data_dir / parquet_file)
if (data_dir / val_samples_file).exists():
    val_samples_df = pd.read_csv(data_dir / val_samples_file)
    val_samples = val_samples_df.sample_id.tolist()
    train_df, val_df = WorldCerealDataset.split_df(df, val_sample_ids=val_samples)
else:
    train_df, val_df = WorldCerealDataset.split_df(df)

logger.info("Setting up model")
if warm_start:
    model_kwargs = json.load(Path(config_dir / "default.json").open("r"))
    model = Presto.load_pretrained(
        valid_month_as_token=model_kwargs["valid_month_as_token"],
        valid_month_size=model_kwargs["valid_month_size"],
    )
    best_model_path: Optional[Path] = default_model_path
else:
    if path_to_config == "":
        path_to_config = config_dir / "default.json"
    model_kwargs = json.load(Path(path_to_config).open("r"))
    model = Presto.construct(**model_kwargs)
    best_model_path = None
model.to(device)

# full finetuning
full_finetuning = WorldCerealFinetuning(train_df, val_df)
finetuned_model = full_finetuning.finetune(model)
model_path = model_logging_dir / Path("models")
model_path.mkdir(exist_ok=True, parents=True)
finetuned_model_path = model_path / "finetuned_model.pt"
torch.save(finetuned_model.state_dict(), finetuned_model_path)

model_modes = ["Random Forest", "Regression", "CatBoostClassifier"]
full_eval = WorldCerealEval(train_df, val_df, spatial_inference_savedir=model_logging_dir)
results = full_eval.finetuning_results_sklearn(
    sklearn_model_modes=model_modes, finetuned_model=finetuned_model
)
logger.info(json.dumps(results, indent=2))

full_maize_eval = WorldCerealEval(
    train_df,
    val_df,
    spatial_inference_savedir=model_logging_dir,
    target_function=target_maize,
    filter_function=filter_remove_noncrops,
    name="WorldCerealMaize",
)
maize_results = full_maize_eval.finetuning_results_sklearn(
    sklearn_model_modes=model_modes, finetuned_model=finetuned_model
)
logger.info(json.dumps(maize_results, indent=2))

# not saving plots to wandb
plot_results(load_world_df(), results, model_logging_dir, show=True, to_wandb=False)
plot_results(
    load_world_df(), maize_results, model_logging_dir, show=True, to_wandb=False, prefix="maize_"
)

# missing data experiments
country_results = []
for country in ["Latvia", "Brazil", "Togo", "Madagascar"]:
    finetuning_task = WorldCerealFinetuning(train_df, val_df, countries_to_remove=[country])
    finetuned_model = finetuning_task.finetune(model)
    for predict_maize in [True, False]:
        kwargs = {
            "train_data": train_df,
            "val_data": val_df,
            "countries_to_remove": [country],
            "spatial_inference_savedir": model_logging_dir,
        }
        if predict_maize:
            kwargs.update(
                {
                    "target_function": target_maize,
                    "filter_function": filter_remove_noncrops,
                    "name": "WorldCerealMaize",
                }
            )
        eval_task = WorldCerealEval(**kwargs)
        results = eval_task.finetuning_results_sklearn(
            finetuned_model=finetuned_model, sklearn_model_modes=model_modes
        )
        logger.info(json.dumps(results, indent=2))
        country_results.append(results)

all_spatial_preds = list(model_logging_dir.glob("*.nc"))
for spatial_preds_path in all_spatial_preds:
    preds = xr.load_dataset(spatial_preds_path)
    output_path = model_logging_dir / f"{spatial_preds_path.stem}.png"
    plot_spatial(preds, output_path, to_wandb=False)

if wandb_enabled:
    wandb.log(results)
    wandb.log(maize_results)
    for results in country_results:
        wandb.log(results)

if wandb_enabled and run:
    run.finish()
    logger.info(f"Wandb url: {run.url}")
    logger.info(f"Wandb url: {run.url}")
