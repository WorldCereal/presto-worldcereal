# presto_pretrain_finetune, but in a notebook
import os.path
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, cast

import pandas as pd
import torch
import xarray as xr

from presto.dataset import (
    WorldCerealBase,
    filter_remove_noncrops,
)
from presto.eval import WorldCerealEval
from presto.presto import Presto, PrestoFineTuningModel
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
argparser.add_argument("--num_workers", type=int, default=64)
argparser.add_argument("--wandb", dest="wandb", action="store_true")
argparser.add_argument("--wandb_org", type=str, default="nasa-harvest")
argparser.add_argument("--parquet_file", type=str, default="rawts-monthly_calval.parquet")

# argparser.add_argument("--val_samples_file", type=str, default="cropland_spatial_generalization_test_split_samples.csv")

argparser.add_argument("--presto_model_type", type=str, default="presto-ft-ct")
argparser.add_argument("--task_type", type=str, default="croptype")
argparser.add_argument("--test_type", type=str, default="random")
argparser.add_argument("--compositing_window", type=str, default="30D")
argparser.add_argument("--time_token", type=str, default="none")

argparser.add_argument("--finetune_classes", type=str, default="CROPTYPE_ALL")
argparser.add_argument("--downstream_classes", type=str, default="CROPTYPE19")

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

presto_model_type: str = args["presto_model_type"]
task_type: str = args["task_type"]
finetune_classes: str = args["finetune_classes"]
downstream_classes: str = args["downstream_classes"]
test_type: str = args["test_type"]
time_token: str = args["time_token"]
assert test_type in ["random", "spatial", "temporal", "seasonal"]

# !!! this can (and should) somehow be inferred from data
compositing_window: str = args["compositing_window"]

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
# val_samples_file: str = args["val_samples_file"]
train_only_samples_file: str = args["train_only_samples_file"]

path_to_config = config_dir / "default.json"
model_kwargs = json.load(Path(path_to_config).open("r"))

model_modes = ["CatBoostClassifier"]
# model_modes = ["Random Forest", "Regression", "CatBoostClassifier"] 

logger.info("Loading data")
df = pd.read_parquet(data_dir / parquet_file)

val_samples_file = f"{task_type}_{test_type}_generalization_test_split_samples.csv"

logger.info(f"Preparing train and val splits for {task_type} {test_type} test")
val_samples_df = pd.read_csv(data_dir / "test_splits" / val_samples_file)

if task_type=="croptype":
    df = WorldCerealBase.map_croptypes(
        df, 
        finetune_classes,
        downstream_classes
        )
    df = filter_remove_noncrops(df)
    
train_df, test_df = WorldCerealBase.split_df(df, val_sample_ids=val_samples_df.sample_id.tolist())

full_eval = WorldCerealEval(
    train_df, test_df, 
    task_type=task_type,
    finetune_classes=finetune_classes
    )

model_path = output_parent_dir / "data" 
model_path.mkdir(exist_ok=True, parents=True)
experiment_prefix = f"{presto_model_type}_{compositing_window}_{task_type}_{test_type}_{time_token}"
finetuned_model_path = model_path / f"{experiment_prefix}.pt"
results_path = model_logging_dir / f"{experiment_prefix}.csv"
catboost_model_path = model_logging_dir / f"{experiment_prefix}.cbm"

# check if finetuned model already exists
logger.info("Checking if the finetuned model exists")
if os.path.isfile(finetuned_model_path):
    logger.info("Finetuned model found! Loading...")
    finetuned_model = Presto.load_pretrained(
        model_path=finetuned_model_path,
        strict=False,
        is_finetuned=True,
        num_outputs=full_eval.num_outputs,
        )
    pretrained_model = Presto.load_pretrained()

    finetuned_model.to(device)
    pretrained_model.to(device)

    results_df_ft = full_eval.evaluate(finetuned_model=finetuned_model, pretrained_model=pretrained_model, croptype_list=full_eval.croptype_list)
    if full_eval.spatial_inference_savedir is not None:
        full_eval.spatial_inference(finetuned_model, None)
    results_df_sklearn, sklearn_models_trained = full_eval.finetuning_results_sklearn(model_modes, finetuned_model)
    results_df_combined = pd.concat([results_df_ft, results_df_sklearn], axis=0)
else:
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
    results_df_combined, finetuned_model, sklearn_models_trained = full_eval.finetuning_results(model, sklearn_model_modes=model_modes)
    torch.save(finetuned_model.state_dict(), finetuned_model_path)

results_df_combined["presto_model_type"] = presto_model_type
results_df_combined["compositing_window"] = compositing_window
results_df_combined["task_type"] = task_type
results_df_combined["test_type"] = test_type
results_df_combined["time_token"] = time_token
results_df_combined.to_csv(results_path, index=False)

for model in sklearn_models_trained:
    model.save_model(catboost_model_path)

all_spatial_preds = list(model_logging_dir.glob("*.nc"))
for spatial_preds_path in all_spatial_preds:
    preds = xr.load_dataset(spatial_preds_path)
    output_path = model_logging_dir / f"{spatial_preds_path.stem}.png"
    plot_spatial(preds, output_path, to_wandb=False)

# if wandb_enabled:
#     wandb.log(results)
#     wandb.log(country_results)
#     wandb.log(year_results)

if wandb_enabled and run:
    run.finish()
    logger.info(f"Wandb url: {run.url}")
