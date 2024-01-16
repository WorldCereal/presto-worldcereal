# presto_pretrain_finetune, but in a notebook
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, cast

import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from presto.dataops import BANDS_GROUPS_IDX
from presto.dataset import WorldCerealMaskedDataset as WorldCerealDataset
from presto.eval import WorldCerealEval
from presto.masking import MASK_STRATEGIES, MaskParamsNoDw
from presto.presto import (
    LossWrapper,
    Presto,
    adjust_learning_rate,
    param_groups_weight_decay,
)
from presto.utils import (
    DEFAULT_SEED,
    config_dir,
    data_dir,
    default_model_path,
    device,
    initialize_logging,
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
argparser.add_argument("--n_epochs", type=int, default=20)
argparser.add_argument("--max_learning_rate", type=float, default=0.0001)
argparser.add_argument("--min_learning_rate", type=float, default=0.0)
argparser.add_argument("--warmup_epochs", type=int, default=2)
argparser.add_argument("--weight_decay", type=float, default=0.05)
argparser.add_argument("--batch_size", type=int, default=4096)
argparser.add_argument("--val_per_n_steps", type=int, default=-1, help="If -1, val every epoch")
argparser.add_argument(
    "--mask_strategies",
    type=str,
    default=[
        "group_bands",
        "random_timesteps",
        "chunk_timesteps",
        "random_combinations",
    ],
    nargs="+",
    help="`all` will use all available masking strategies (including single bands)",
)
argparser.add_argument("--mask_ratio", type=float, default=0.75)
argparser.add_argument("--seed", type=int, default=DEFAULT_SEED)
argparser.add_argument("--num_workers", type=int, default=4)
argparser.add_argument("--wandb", dest="wandb", action="store_true")
argparser.add_argument("--wandb_org", type=str, default="nasa-harvest")
argparser.add_argument(
    "--train_file",
    type=str,
    default="worldcereal_presto_cropland_nointerp_V1_TRAIN.parquet",
)
argparser.add_argument(
    "--val_file",
    type=str,
    default="worldcereal_presto_cropland_nointerp_V2_VAL.parquet",
)
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

num_epochs = args["n_epochs"]
val_per_n_steps = args["val_per_n_steps"]
max_learning_rate = args["max_learning_rate"]
min_learning_rate = args["min_learning_rate"]
warmup_epochs = args["warmup_epochs"]
weight_decay = args["weight_decay"]
batch_size = args["batch_size"]

# Default mask strategies and mask_ratio
mask_strategies: Tuple[str, ...] = tuple(args["mask_strategies"])
if (len(mask_strategies) == 1) and (mask_strategies[0] == "all"):
    mask_strategies = MASK_STRATEGIES
mask_ratio: float = args["mask_ratio"]

train_file: str = args["train_file"]
val_file: str = args["val_file"]

path_to_config = config_dir / "default.json"
model_kwargs = json.load(Path(path_to_config).open("r"))

logger.info("Setting up dataloaders")

# Load the mask parameters
mask_params = MaskParamsNoDw(mask_strategies, mask_ratio)

train_df = pd.read_parquet(data_dir / train_file)
val_df = pd.read_parquet(data_dir / val_file)
train_dataloader = DataLoader(
    WorldCerealDataset(train_df, mask_params=mask_params),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
)
val_dataloader = DataLoader(
    WorldCerealDataset(val_df, mask_params=mask_params),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)
validation_task = WorldCerealEval(
    train_data=train_df.sample(1000, random_state=DEFAULT_SEED),
    val_data=val_df.sample(1000, random_state=DEFAULT_SEED),
)

if val_per_n_steps == -1:
    val_per_n_steps = len(train_dataloader)

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

param_groups = param_groups_weight_decay(model, weight_decay)
optimizer = optim.AdamW(param_groups, lr=max_learning_rate, betas=(0.9, 0.95))
mse = LossWrapper(nn.MSELoss())

training_config = {
    "model": model.__class__,
    "encoder": model.encoder.__class__,
    "decoder": model.decoder.__class__,
    "optimizer": optimizer.__class__.__name__,
    "eo_loss": mse.loss.__class__.__name__,
    "device": device,
    "logging_dir": model_logging_dir,
    **args,
    **model_kwargs,
}

if wandb_enabled:
    wandb.config.update(training_config)

lowest_validation_loss = None
best_val_epoch = 0
training_step = 0
num_validations = 0

with tqdm(range(num_epochs), desc="Epoch") as tqdm_epoch:
    for epoch in tqdm_epoch:
        # ------------------------ Training ----------------------------------------
        total_eo_train_loss = 0.0
        num_updates_being_captured = 0
        train_size = 0
        model.train()
        for epoch_step, b in enumerate(tqdm(train_dataloader, desc="Train", leave=False)):
            mask, x, y, start_month = b[0].to(device), b[2].to(device), b[3].to(device), b[6]
            dw_mask, x_dw, y_dw = b[1].to(device), b[4].to(device).long(), b[5].to(device).long()
            latlons, real_mask = b[7].to(device), b[9].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            lr = adjust_learning_rate(
                optimizer,
                epoch_step / len(train_dataloader) + epoch,
                warmup_epochs,
                num_epochs,
                max_learning_rate,
                min_learning_rate,
            )
            # Get model outputs and calculate loss
            y_pred, dw_pred = model(
                x, mask=mask, dynamic_world=x_dw, latlons=latlons, month=start_month
            )
            # set all SRTM timesteps except the first one to unmasked, so that
            # they will get ignored by the loss function even if the SRTM
            # value was masked
            mask[:, 1:, BANDS_GROUPS_IDX["SRTM"]] = False
            # set the "truly masked" values to unmasked, so they also get ignored in the loss
            mask[real_mask] = False
            loss = mse(y_pred[mask], y[mask])
            loss.backward()
            optimizer.step()

            current_batch_size = len(x)
            total_eo_train_loss += loss.item()
            num_updates_being_captured += 1
            train_size += current_batch_size
            training_step += 1

            # ------------------------ Validation --------------------------------------
            if training_step % val_per_n_steps == 0:
                total_eo_val_loss = 0.0
                num_val_updates_captured = 0
                val_size = 0
                model.eval()

                with torch.no_grad():
                    for b in tqdm(val_dataloader, desc="Validate"):
                        mask, x, y, start_month, real_mask = (
                            b[0].to(device),
                            b[2].to(device),
                            b[3].to(device),
                            b[6],
                            b[9].to(device),
                        )
                        dw_mask, x_dw = b[1].to(device), b[4].to(device).long()
                        y_dw, latlons = b[5].to(device).long(), b[7].to(device)
                        # Get model outputs and calculate loss
                        y_pred, dw_pred = model(
                            x, mask=mask, dynamic_world=x_dw, latlons=latlons, month=start_month
                        )
                        # set all SRTM timesteps except the first one to unmasked, so that
                        # they will get ignored by the loss function even if the SRTM
                        # value was masked
                        mask[:, 1:, BANDS_GROUPS_IDX["SRTM"]] = False
                        # set the "truly masked" values to unmasked, so they also get
                        # ignored in the loss
                        mask[real_mask] = False
                        loss = mse(y_pred[mask], y[mask])
                        current_batch_size = len(x)
                        total_eo_val_loss += loss.item()
                        num_val_updates_captured += 1

                # ------------------------ Metrics + Logging -------------------------------
                # train_loss now reflects the value against which we calculate gradients
                train_eo_loss = total_eo_train_loss / num_updates_being_captured
                val_eo_loss = total_eo_val_loss / num_val_updates_captured

                if "train_size" not in training_config and "val_size" not in training_config:
                    training_config["train_size"] = train_size
                    training_config["val_size"] = val_size
                    if wandb_enabled:
                        wandb.config.update(training_config)

                to_log = {
                    "train_eo_loss": train_eo_loss,
                    "val_eo_loss": val_eo_loss,
                    "training_step": training_step,
                    "epoch": epoch,
                    "lr": lr,
                }
                tqdm_epoch.set_postfix(loss=val_eo_loss)

                val_task_results, _ = validation_task.finetuning_results(
                    model, sklearn_model_modes=["Random Forest"]
                )
                to_log.update(val_task_results)

                if lowest_validation_loss is None or val_eo_loss < lowest_validation_loss:
                    lowest_validation_loss = val_eo_loss
                    best_val_epoch = epoch

                    model_path = model_logging_dir / Path("models")
                    model_path.mkdir(exist_ok=True, parents=True)

                    best_model_path = model_path / f"{model_name}{epoch}.pt"
                    logger.info(f"Saving best model to: {best_model_path}")
                    torch.save(model.state_dict(), best_model_path)

                # reset training logging
                total_eo_train_loss = 0.0
                num_updates_being_captured = 0
                train_size = 0
                num_validations += 1

                if wandb_enabled:
                    wandb.log(to_log)

                model.train()

logger.info(f"Trained for {num_epochs} epochs, best model at {best_model_path}")

if best_model_path is not None:
    logger.info("Loading best model: %s" % best_model_path)
    best_model = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_model)
else:
    logger.info("Running eval with randomly init weights")

full_eval = WorldCerealEval(train_df, val_df, model_logging_dir)
results, finetuned_model = full_eval.finetuning_results(
    model, sklearn_model_modes=["Random Forest", "Regression", "CatBoostClassifier"]
)
if finetuned_model is not None:
    model_path = model_logging_dir / Path("models")
    model_path.mkdir(exist_ok=True, parents=True)
    finetuned_model_path = model_path / "finetuned_model.pt"
    torch.save(model.state_dict(), finetuned_model_path)
plot_results(full_eval.world_df, results, model_logging_dir, show=True, to_wandb=wandb_enabled)
all_spatial_preds = list(model_logging_dir.glob("*.nc"))
for spatial_preds_path in all_spatial_preds:
    preds = xr.load_dataset(spatial_preds_path)
    output_path = model_logging_dir / f"{spatial_preds_path.stem}.png"
    plot_spatial(preds, output_path, to_wandb=wandb_enabled)


logger.info(json.dumps(results, indent=2))
if wandb_enabled:
    wandb.log(results)

if wandb_enabled and run:
    run.finish()
    logger.info(f"Wandb url: {run.url}")
    logger.info(f"Wandb url: {run.url}")
