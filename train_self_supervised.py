# presto_pretrain_finetune, but in a notebook
import argparse
import json
import logging

# import os.path
# import pickle
from pathlib import Path
from typing import Optional, Tuple, cast

import pandas as pd
import requests
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# import xarray as xr
from presto.dataops import BANDS_GROUPS_IDX
from presto.dataset import WorldCerealBase, WorldCerealMaskedDataset
from presto.eval import WorldCerealEval
from presto.masking import MASK_STRATEGIES, MaskParamsNoDw
from presto.presto import (
    LossWrapper,
    Presto,
    adjust_learning_rate,
    extend_to_dekadal,
    param_groups_weight_decay,
)
from presto.utils import (  # plot_spatial,
    DEFAULT_SEED,
    config_dir,
    data_dir,
    default_model_path,
    device,
    initialize_logging,
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
argparser.add_argument(
    "--parquet_file",
    type=str,
    default="rawts-monthly_calval.parquet",
    choices=["rawts-monthly_calval.parquet", "rawts-10d_calval.parquet"],
)
argparser.add_argument("--n_epochs", type=int, default=20)
argparser.add_argument("--max_learning_rate", type=float, default=0.0001)
argparser.add_argument("--min_learning_rate", type=float, default=0.0)
argparser.add_argument("--finetune_train_masking", type=float, default=0.0)
argparser.add_argument("--warmup_epochs", type=int, default=2)
argparser.add_argument("--weight_decay", type=float, default=0.05)
argparser.add_argument("--batch_size", type=int, default=2048)
argparser.add_argument("--val_per_n_steps", type=int, default=-1, help="If -1, val every epoch")
argparser.add_argument("--mask_ratio", type=float, default=0.75)
argparser.add_argument(
    "--test_type",
    type=str,
    default="random",
    choices=["random", "spatial", "temporal", "seasonal"],
)
argparser.add_argument("--train_only_samples_file", type=str, default="train_only_samples.csv")

argparser.add_argument("--warm_start", dest="warm_start", action="store_true")
argparser.set_defaults(wandb=False)
argparser.set_defaults(warm_start=True)
args = argparser.parse_args().__dict__

model_name = args["model_name"]
seed = args["seed"]
num_workers = args["num_workers"]
num_epochs = args["n_epochs"]
val_per_n_steps = args["val_per_n_steps"]
max_learning_rate = args["max_learning_rate"]
min_learning_rate = args["min_learning_rate"]
warmup_epochs = args["warmup_epochs"]
weight_decay = args["weight_decay"]
batch_size = args["batch_size"]
path_to_config = args["path_to_config"]
warm_start = args["warm_start"]
wandb_enabled = args["wandb"]
wandb_org = args["wandb_org"]

# Default mask strategies and mask_ratio
mask_strategies: Tuple[str, ...] = tuple(args["mask_strategies"])
if (len(mask_strategies) == 1) and (mask_strategies[0] == "all"):
    mask_strategies = MASK_STRATEGIES
mask_ratio: float = args["mask_ratio"]

presto_model_description: str = args["presto_model_description"]
test_type: str = args["test_type"]
assert test_type in ["random", "spatial", "temporal", "seasonal"]

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
train_only_samples_file: str = args["train_only_samples_file"]

dekadal = False
compositing_window = "30D"
if "10d" in parquet_file:
    dekadal = True
    compositing_window = "10D"

valid_month_as_token = False

path_to_config = config_dir / "default.json"
with open(path_to_config) as file:
    model_kwargs = json.load(file)


logger.info("Loading data")
df = pd.read_parquet(data_dir / parquet_file)

val_samples_file = f"cropland_{test_type}_generalization_test_split_samples.csv"

logger.info(f"Preparing train and val splits for {test_type} test")
val_samples_df = pd.read_csv(data_dir / "test_splits" / val_samples_file)

train_df, val_df = WorldCerealBase.split_df(df, val_sample_ids=val_samples_df.sample_id.tolist())


# Load the mask parameters
mask_params = MaskParamsNoDw(mask_strategies, mask_ratio, num_timesteps=36 if dekadal else 12)
masked_ds = WorldCerealMaskedDataset

train_dataloader = DataLoader(
    masked_ds(train_df, mask_params=mask_params),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
)
val_dataloader = DataLoader(
    masked_ds(val_df, mask_params=mask_params),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)
validation_task = WorldCerealEval(
    train_data=train_df.sample(1000, random_state=DEFAULT_SEED),
    test_data=val_df.sample(1000, random_state=DEFAULT_SEED),
)


if val_per_n_steps == -1:
    val_per_n_steps = len(train_dataloader)

logger.info("Setting up model")
if warm_start:

    warm_start_model_name = "presto-pt"
    warm_start_model_path = f"""
    https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/{warm_start_model_name}_{compositing_window}.pt
    """

    if requests.get(warm_start_model_path).status_code >= 400:
        logger.error(f"No url for {warm_start_model_name} available")

    model = Presto.load_pretrained(
        model_path=warm_start_model_path,
        from_url=True,
        dekadal=dekadal,
        valid_month_as_token=False,
        strict=False,
    )
    best_model_path: Optional[Path] = default_model_path
else:
    if path_to_config == "":
        path_to_config = config_dir / "default.json"
    model_kwargs = json.load(Path(path_to_config).open("r"))
    model = Presto.construct(**model_kwargs)
    best_model_path = None

if dekadal:
    logger.info("extending model to dekadal architecture")
    model = extend_to_dekadal(model)
model.to(device)
# print(f"model pos embed shape {model.encoder.pos_embed.shape}") # correctly reinitialized

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
            mask, x, y, start_month, valid_month = (
                b[0].to(device),
                b[2].to(device),
                b[3].to(device),
                b[6].to(device),
                b[10].to(device),
            )
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
                x,
                mask=mask,
                dynamic_world=x_dw,
                latlons=latlons,
                month=start_month,
                valid_month=valid_month,
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
                        mask, x, y, start_month, real_mask, valid_month = (
                            b[0].to(device),
                            b[2].to(device),
                            b[3].to(device),
                            b[6].to(device),
                            b[9].to(device),
                            b[10].to(device),
                        )
                        dw_mask, x_dw = b[1].to(device), b[4].to(device).long()
                        y_dw, latlons = b[5].to(device).long(), b[7].to(device)
                        # Get model outputs and calculate loss
                        y_pred, dw_pred = model(
                            x,
                            mask=mask,
                            dynamic_world=x_dw,
                            latlons=latlons,
                            month=start_month,
                            valid_month=valid_month,
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
