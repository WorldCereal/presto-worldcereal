import logging
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import utils
from .dataset import WorldCerealLabelledDataset
from .presto import Presto, PrestoFineTuningModel, param_groups_lrd
from .utils import DEFAULT_SEED, device

logger = logging.getLogger("__main__")

world_shp_path = "world-administrative-boundaries/world-administrative-boundaries.shp"


@dataclass
class Hyperparams:
    lr: float = 2e-5
    max_epochs: int = 20
    batch_size: int = 64
    patience: int = 3
    num_workers: int = 4


class WorldCerealEval:
    name = "WorldCerealCropland"
    threshold = 0.5
    num_outputs = 1
    regression = False

    def __init__(self, train_data: pd.DataFrame, val_data: pd.DataFrame, seed: int = DEFAULT_SEED):
        self.seed = seed

        # SAR cannot equal 0.0 since we take the log of it
        cols = [f"SAR-{s}-ts{t}-20m" for s in ["VV", "VH"] for t in range(12)]
        self.train_df = train_data[~(train_data.loc[:, cols] == 0.0).any(axis=1)]

        self.val_df = val_data.drop_duplicates(subset=["pixelids", "lat", "lon", "end_date"])
        self.val_df = self.val_df[~pd.isna(self.val_df).any(axis=1)]
        self.val_df = self.val_df[~(self.val_df.loc[:, cols] == 0.0).any(axis=1)]
        self.test_df = self.val_df

        self.world_df = gpd.read_file(utils.data_dir / world_shp_path)
        # these columns contain nan sometimes
        self.world_df = self.world_df.drop(columns=["iso3", "status", "color_code", "iso_3166_1_"])

    def _construct_finetuning_model(self, pretrained_model: Presto) -> PrestoFineTuningModel:
        model = cast(Callable, pretrained_model.construct_finetuning_model)(
            num_outputs=self.num_outputs
        )
        return model

    @torch.no_grad()
    def finetune_sklearn_model(
        self,
        dl: DataLoader,
        pretrained_model,
        models: List[str] = ["Regression", "Random Forest"],
    ) -> Union[Sequence[BaseEstimator], Dict]:
        for model_mode in models:
            assert model_mode in ["Regression", "Random Forest"]
        pretrained_model.eval()

        encoding_list, target_list = [], []
        for x, y, dw, latlons, month, _, variable_mask in dl:
            x_f, dw_f, latlons_f, month_f, variable_mask_f = [
                t.to(device) for t in (x, dw, latlons, month, variable_mask)
            ]
            target_list.append(y)
            with torch.no_grad():
                encodings = (
                    pretrained_model.encoder(
                        x_f,
                        dynamic_world=dw_f.long(),
                        mask=variable_mask_f,
                        latlons=latlons_f,
                        month=month_f,
                    )
                    .cpu()
                    .numpy()
                )
                encoding_list.append(encodings)
        encodings_np = np.concatenate(encoding_list)
        targets = np.concatenate(target_list)
        if len(targets.shape) == 2 and targets.shape[1] == 1:
            targets = targets.ravel()

        fit_models = []
        model_dict = {
            "Regression": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=self.seed
            ),
            "Random Forest": RandomForestClassifier(
                class_weight="balanced", random_state=self.seed
            ),
        }
        for model in tqdm(models, desc="Fitting sklearn models"):
            fit_models.append(clone(model_dict[model]).fit(encodings_np, targets))
        return fit_models

    @torch.no_grad()
    def evaluate(
        self,
        finetuned_model: Union[PrestoFineTuningModel, BaseEstimator],
        pretrained_model: Optional[Presto] = None,
    ) -> Dict:
        if isinstance(finetuned_model, BaseEstimator):
            assert isinstance(pretrained_model, Presto)

        test_ds = WorldCerealLabelledDataset(self.test_df)
        dl = DataLoader(
            test_ds,
            batch_size=8192,
            shuffle=False,  # keep as False!
            num_workers=4,
        )
        assert isinstance(dl.sampler, torch.utils.data.SequentialSampler)

        test_preds, targets = [], []

        for x, y, dw, latlons, month, num_masked_tokens, variable_mask in dl:
            targets.append(y)
            x_f, dw_f, latlons_f, month_f, variable_mask_f = [
                t.to(device) for t in (x, dw, latlons, month, variable_mask)
            ]
            if isinstance(finetuned_model, PrestoFineTuningModel):
                finetuned_model.eval()
                preds = finetuned_model(
                    x_f,
                    dynamic_world=dw_f.long(),
                    mask=variable_mask_f,
                    latlons=latlons_f,
                    month=month_f,
                ).squeeze(dim=1)
                preds = torch.sigmoid(preds).cpu().numpy()
            elif isinstance(finetuned_model, BaseEstimator):
                cast(Presto, pretrained_model).eval()
                encodings = (
                    cast(Presto, pretrained_model)
                    .encoder(
                        x_f,
                        dynamic_world=dw_f.long(),
                        mask=variable_mask_f,
                        latlons=latlons_f,
                        month=month_f,
                    )
                    .cpu()
                    .numpy()
                )
                preds = finetuned_model.predict(encodings)
            test_preds.append(preds)

        test_preds_np = np.concatenate(test_preds) >= self.threshold
        target_np = np.concatenate(targets)
        prefix = f"{self.name}_{finetuned_model.__class__.__name__}"

        test_df = self.test_df.loc[
            ~self.test_df.LANDCOVER_LABEL.isin(WorldCerealLabelledDataset.FILTER_LABELS)
        ]
        catboost_preds = test_df.catboost_prediction

        def format_partitioned(results):
            return {
                "{p}_{m}".format(p=self.name if "CatBoost" in m else prefix, m=m): float(val)
                for (m, val) in results.items()
            }

        return {
            f"{prefix}_f1": float(f1_score(target_np, test_preds_np)),
            f"{prefix}_recall": float(recall_score(target_np, test_preds_np)),
            f"{prefix}_precision": float(precision_score(target_np, test_preds_np)),
            f"{self.name}_CatBoost_f1": float(f1_score(target_np, catboost_preds)),
            f"{self.name}_CatBoost_recall": float(recall_score(target_np, catboost_preds)),
            f"{self.name}_CatBoost_precision": float(precision_score(target_np, catboost_preds)),
            **format_partitioned(self.partitioned_metrics(target_np, test_preds_np)),
        }

    @staticmethod
    def metrics(
        prefix: str, prop_series: pd.Series, preds: np.ndarray, target: np.ndarray
    ) -> Dict:
        res = {}
        precisions, recalls = [], []
        for prop in prop_series.dropna().unique():
            f: pd.Series = cast(pd.Series, prop_series == prop)
            # Recall (and hence F1) are nan iff there are no ground-truth positives
            recall = recall_score(target[f], preds[f], zero_division=np.nan)
            precision = precision_score(target[f], preds[f], zero_division=0.0)
            recalls.append(recall)
            precisions.append(precision)
            res.update(
                {
                    f"{prefix}_num_samples: {prop}": f.sum(),
                    f"{prefix}_num_positives: {prop}": target[f].sum(),
                    f"{prefix}_num_predicted: {prop}": preds[f].sum(),
                    # +1e-6 to avoid ZeroDivisionError and be 0.0 instead
                    f"{prefix}_f1: {prop}": 2 * recall * precision / (precision + recall + 1e-6),
                    f"{prefix}_recall: {prop}": recall,
                    f"{prefix}_precision: {prop}": precision,
                }
            )
        recall, precision = np.nanmean(recalls), np.nanmean(precisions)
        res.update(
            {
                f"{prefix}_f1: macro": 2 * recall * precision / (precision + recall + 1e-6),
                f"{prefix}_recall: macro": recall,
                f"{prefix}_precision: macro": precision,
            }
        )
        return res

    def partitioned_metrics(
        self, target: np.ndarray, preds: np.ndarray
    ) -> Dict[str, Union[np.float32, np.int32]]:
        test_df = self.test_df.loc[
            ~self.test_df.LANDCOVER_LABEL.isin(WorldCerealLabelledDataset.FILTER_LABELS)
        ]
        catboost_preds = test_df.catboost_prediction
        years = test_df.end_date.apply(lambda date: date[:4])

        latlons = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries.from_xy(x=test_df.lon, y=test_df.lat), crs="EPSG:4326"
        )
        # project to non geographic CRS, otherwise geopandas gives a warning
        world_attrs = gpd.sjoin_nearest(
            latlons.to_crs("EPSG:3857"), self.world_df.to_crs("EPSG:3857"), how="left"
        )
        world_attrs = world_attrs[~world_attrs.index.duplicated(keep="first")]
        if world_attrs.isna().any(axis=1).any():
            logger.warning("Some coordinates couldn't be matched to a country")

        metrics = partial(self.metrics, target=target)
        return {
            **metrics("aez", test_df.aez_zoneid, preds),
            **metrics("year", years, preds),
            **metrics("country", world_attrs.name, preds),
            **metrics("continent", world_attrs.continent, preds),
            **metrics("region", world_attrs.region, preds),
            **metrics("CatBoost_aez", test_df.aez_zoneid, catboost_preds),
            **metrics("CatBoost_year", years, catboost_preds),
            **metrics("CatBoost_country", world_attrs.name, catboost_preds),
            **metrics("CatBoost_continent", world_attrs.continent, catboost_preds),
            **metrics("CatBoost_region", world_attrs.region, catboost_preds),
        }

    def finetune(self, pretrained_model) -> PrestoFineTuningModel:
        hyperparams = Hyperparams()
        model = self._construct_finetuning_model(pretrained_model)

        parameters = param_groups_lrd(model)
        optimizer = AdamW(parameters, lr=hyperparams.lr)

        train_ds = WorldCerealLabelledDataset(self.train_df)
        val_ds = WorldCerealLabelledDataset(self.val_df)

        pos = (train_ds.df.LANDCOVER_LABEL == 11).sum()
        wts = 1 / torch.tensor([len(train_ds.df) - pos, pos])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=(wts / wts[0])[1])

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        train_dl = DataLoader(
            train_ds,
            batch_size=hyperparams.batch_size,
            shuffle=True,
            num_workers=hyperparams.num_workers,
            generator=generator,
        )

        val_dl = DataLoader(
            val_ds,
            batch_size=hyperparams.batch_size,
            shuffle=False,
            num_workers=hyperparams.num_workers,
        )

        train_loss = []
        val_loss = []
        best_loss = None
        best_model_dict = None
        epochs_since_improvement = 0

        run = None
        try:
            import wandb

            run = wandb.run
        except ImportError:
            pass

        for _ in tqdm(range(hyperparams.max_epochs), desc="Finetuning"):
            model.train()
            epoch_train_loss = 0.0
            for x, y, dw, latlons, month, _, variable_mask in tqdm(
                train_dl, desc="Training", leave=False
            ):
                x, y, dw, latlons, month, variable_mask = [
                    t.to(device) for t in (x, y, dw, latlons, month, variable_mask)
                ]
                optimizer.zero_grad()
                preds = model(
                    x,
                    dynamic_world=dw.long(),
                    mask=variable_mask,
                    latlons=latlons,
                    month=month,
                )
                loss = loss_fn(preds.squeeze(-1), y.float())
                epoch_train_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss.append(epoch_train_loss / len(train_dl))

            model.eval()
            all_preds, all_y = [], []
            for x, y, dw, latlons, month, _, variable_mask in val_dl:
                x, y, dw, latlons, month, variable_mask = [
                    t.to(device) for t in (x, y, dw, latlons, month, variable_mask)
                ]
                with torch.no_grad():
                    preds = model(
                        x,
                        dynamic_world=dw.long(),
                        mask=variable_mask,
                        latlons=latlons,
                        month=month,
                    )
                    all_preds.append(preds.squeeze(-1))
                    all_y.append(y.float())

            val_loss.append(loss_fn(torch.cat(all_preds), torch.cat(all_y)))

            if run is not None:
                wandb.log(
                    {
                        f"{self.name}_finetuning_val_loss": val_loss[-1],
                        f"{self.name}_finetuning_train_loss": train_loss[-1],
                    }
                )

            if best_loss is None:
                best_loss = val_loss[-1]
                best_model_dict = deepcopy(model.state_dict())
            else:
                if val_loss[-1] < best_loss:
                    best_loss = val_loss[-1]
                    best_model_dict = deepcopy(model.state_dict())
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    if epochs_since_improvement >= hyperparams.patience:
                        logger.info("Early stopping!")
                        break
        assert best_model_dict is not None
        model.load_state_dict(best_model_dict)

        model.eval()
        return model

    def finetuning_results(
        self,
        pretrained_model,
        model_modes: List[str],
    ) -> Tuple[Dict, Optional[PrestoFineTuningModel]]:
        for model_mode in model_modes:
            assert model_mode in ["Regression", "Random Forest", "finetune"]

        results_dict = {}
        finetuned_model: Optional[PrestoFineTuningModel] = None
        if "finetune" in model_modes:
            finetuned_model = self.finetune(pretrained_model)
            results_dict.update(self.evaluate(finetuned_model, None))

        sklearn_modes = [x for x in model_modes if x != "finetune"]
        if len(sklearn_modes) > 0:
            dl = DataLoader(
                WorldCerealLabelledDataset(self.train_df),
                batch_size=2048,
                shuffle=False,
                num_workers=4,
            )
            sklearn_models = self.finetune_sklearn_model(
                dl,
                pretrained_model,
                models=sklearn_modes,
            )
            for sklearn_model in sklearn_models:
                logger.info(f"Evaluating {sklearn_model}...")
                results_dict.update(self.evaluate(sklearn_model, pretrained_model))
        return results_dict, finetuned_model
