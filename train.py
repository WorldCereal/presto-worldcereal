# presto_pretrain_finetune, but in a notebook
import json
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataops import BANDS_GROUPS_IDX
from src.dataset import WorldCerealDataset
from src.masking import MASK_STRATEGIES, MaskParamsNoDw
from src.presto import (
    LossWrapper,
    Presto,
    adjust_learning_rate,
    param_groups_weight_decay,
)
from src.utils import (
    DEFAULT_SEED,
    config_dir,
    data_dir,
    device,
    initialize_logging,
    seed_everything,
    timestamp_dirname,
)

logger = logging.getLogger("__main__")


model_name = "presto_worldcereal"
seed = DEFAULT_SEED
seed_everything(seed)
output_parent_dir = Path(".")
run_id = None

logging_dir = output_parent_dir / "output" / timestamp_dirname(run_id)
logging_dir.mkdir(exist_ok=True, parents=True)
initialize_logging(logging_dir)
logger.info("Using output dir: %s" % logging_dir)

# Taken the defaults for now
num_epochs = 20
val_per_n_steps = 1000
max_learning_rate = 0.0001  # 0.001 is default, for finetuning max should be lower?
min_learning_rate = 0
warmup_epochs = 2
weight_decay = 0.05
batch_size = 4096  # default 4096

# Default mask strategies and mask_ratio
mask_strategies = MASK_STRATEGIES
mask_ratio: float = 0.75

path_to_config = config_dir / "default.json"
model_kwargs = json.load(Path(path_to_config).open("r"))

logger.info("Setting up dataloaders")

# Load the mask parameters
mask_params = MaskParamsNoDw(mask_strategies, mask_ratio)

# Create DataLoaders from the dataset. For now, without shame using same data for train and val
# we're just testing functionality ;-)
df = pd.read_parquet(data_dir / "worldcereal_testdf.parquet")
# Create the WorldCereal dataset
ds = WorldCerealDataset(df, mask_params=mask_params)
train_dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)


logger.info("Setting up model")
model = Presto.load_pretrained()
model.to(device)

# Model hyperparameters: keep unchanged for now
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
    "logging_dir": logging_dir,
    # **args,
    # **model_kwargs,
}

lowest_validation_loss = None
best_val_epoch = 0
training_step = 0
num_validations = 0
dataloader_length = df.shape[0]

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
            latlons = b[7].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            lr = adjust_learning_rate(
                optimizer,
                epoch_step / dataloader_length + epoch,
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
                        mask, x, y, start_month = (
                            b[0].to(device),
                            b[2].to(device),
                            b[3].to(device),
                            b[6],
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
                    # if wandb_enabled:
                    #     wandb.config.update(training_config)

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

                    model_path = logging_dir / Path("models")
                    model_path.mkdir(exist_ok=True, parents=True)

                    best_model_path = model_path / f"{model_name}{epoch}.pt"
                    logger.info(f"Saving best model to: {best_model_path}")
                    torch.save(model.state_dict(), best_model_path)

                # reset training logging
                total_eo_train_loss = 0.0
                num_updates_being_captured = 0
                train_size = 0
                num_validations += 1

                # if wandb_enabled:
                #     model.eval()
                #     for title, plot in plot_predictions(model):
                #         to_log[title] = plot
                #     wandb.log(to_log)
                #     plt.close("all")

                model.train()

logger.info(f"Done training, best model saved to {best_model_path}")
