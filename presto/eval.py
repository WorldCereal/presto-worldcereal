import logging
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .dataset import (
    NORMED_BANDS,
    WorldCerealInferenceDataset,
    WorldCerealLabelled10DDataset,
    WorldCerealLabelledDataset,
)
from .presto import (
    Presto,
    PrestoFineTuningModel,
    get_sinusoid_encoding_table,
    param_groups_lrd,
)
from .utils import DEFAULT_SEED, device, prep_dataframe

logger = logging.getLogger("__main__")

SklearnStyleModel = Union[BaseEstimator, CatBoostClassifier]


@dataclass
class Hyperparams:
    lr: float = 2e-5
    max_epochs: int = 100
    batch_size: int = 64
    patience: int = 10
    num_workers: int = 4


class WorldCerealEval:
    name = "WorldCerealCropland"
    threshold = 0.5
    num_outputs = 1
    regression = False

    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        countries_to_remove: Optional[List[str]] = None,
        years_to_remove: Optional[List[int]] = None,
        spatial_inference_savedir: Optional[Path] = None,
        seed: int = DEFAULT_SEED,
        target_function: Optional[Callable[[Dict], int]] = None,
        filter_function: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        name: Optional[str] = None,
        val_size: float = 0.2,
        dekadal: bool = False,
        train_masking: float = 0.0,
    ):
        self.seed = seed

        if name is not None:
            self.name = name
        self.target_function = target_function

        train_data, val_data = WorldCerealLabelledDataset.split_df(train_data, val_size=val_size)
        self.train_df = prep_dataframe(train_data, filter_function, dekadal=dekadal)
        self.val_df = prep_dataframe(val_data, filter_function, dekadal=dekadal)
        self.test_df = prep_dataframe(test_data, filter_function, dekadal=dekadal)

        self.spatial_inference_savedir = spatial_inference_savedir

        self.countries_to_remove = countries_to_remove
        self.years_to_remove = years_to_remove

        if self.countries_to_remove is not None:
            self.name = f"{self.name}_removed_countries_{countries_to_remove}"
        if self.years_to_remove is not None:
            self.name = f"{self.name}_removed_years_{years_to_remove}"

        self.dekadal = dekadal
        self.ds_class = WorldCerealLabelled10DDataset if dekadal else WorldCerealLabelledDataset
        self.train_masking = train_masking

    def _construct_finetuning_model(self, pretrained_model: Presto) -> PrestoFineTuningModel:
        model: PrestoFineTuningModel = cast(Callable, pretrained_model.construct_finetuning_model)(
            num_outputs=self.num_outputs
        )

        if self.dekadal:
            max_sequence_length = 72  # can this be 36?
            old_pos_embed_device = model.encoder.pos_embed.device
            model.encoder.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    max_sequence_length,
                    model.encoder.pos_embed.shape[-1],
                    device=old_pos_embed_device,
                ),
                requires_grad=False,
            )
            pos_embed = get_sinusoid_encoding_table(
                model.encoder.pos_embed.shape[1], model.encoder.pos_embed.shape[-1]
            )
            model.encoder.pos_embed.data.copy_(pos_embed.to(device=old_pos_embed_device))
        return model

    @torch.no_grad()
    def finetune_sklearn_model(
        self,
        dl: DataLoader,
        val_dl: DataLoader,
        pretrained_model: PrestoFineTuningModel,
        models: List[str] = ["Regression", "Random Forest"],
    ) -> Union[Sequence[BaseEstimator], Dict]:
        for model_mode in models:
            assert model_mode in ["Regression", "Random Forest", "CatBoostClassifier"]
        pretrained_model.eval()

        def dataloader_to_encodings_and_targets(
            dl: DataLoader,
        ) -> Tuple[np.ndarray, np.ndarray]:
            encoding_list, target_list = [], []
            for x, y, dw, latlons, month, variable_mask in dl:
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
            return encodings_np, targets

        train_encodings, train_targets = dataloader_to_encodings_and_targets(dl)
        val_encodings, val_targets = dataloader_to_encodings_and_targets(val_dl)

        fit_models = []
        class_weights = cast(WorldCerealLabelledDataset, dl.dataset).class_weights
        class_weight_dict = {idx: weight for idx, weight in enumerate(class_weights)}
        model_dict = {
            "Regression": LogisticRegression(
                class_weight=class_weight_dict,
                max_iter=1000,
                random_state=self.seed,
            ),
            "Random Forest": RandomForestClassifier(
                class_weight=class_weight_dict,
                random_state=self.seed,
            ),
            # Parameters emulate
            # https://github.com/WorldCereal/wc-classification/blob/
            # 4a9a839507d9b4f63c378b3b1d164325cbe843d6/src/worldcereal/classification/models.py#L490
            "CatBoostClassifier": CatBoostClassifier(
                iterations=8000,
                depth=8,
                learning_rate=0.05,
                early_stopping_rounds=20,
                l2_leaf_reg=3,
                random_state=self.seed,
                class_weights=class_weight_dict,
            ),
        }
        for model in tqdm(models, desc="Fitting sklearn models"):
            if model == "CatBoostClassifier":
                fit_models.append(
                    clone(model_dict[model]).fit(
                        train_encodings,
                        train_targets,
                        eval_set=Pool(val_encodings, val_targets),
                    )
                )
            else:
                fit_models.append(clone(model_dict[model]).fit(train_encodings, train_targets))
        return fit_models

    @staticmethod
    def _inference_for_dl(
        dl,
        finetuned_model: Union[PrestoFineTuningModel, SklearnStyleModel],
        pretrained_model: Optional[PrestoFineTuningModel] = None,
    ) -> Tuple:

        test_preds, targets = [], []

        for x, y, dw, latlons, month, variable_mask in dl:
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
            else:
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
                preds = finetuned_model.predict_proba(encodings)[:, 1]
            test_preds.append(preds)

        test_preds_np = np.concatenate(test_preds)
        target_np = np.concatenate(targets)
        return test_preds_np, target_np

    @torch.no_grad()
    def spatial_inference(
        self,
        finetuned_model: Union[PrestoFineTuningModel, SklearnStyleModel],
        pretrained_model: Optional[PrestoFineTuningModel] = None,
    ):
        assert self.spatial_inference_savedir is not None
        ds = WorldCerealInferenceDataset()
        for i in range(len(ds)):
            (
                eo,
                dynamic_world,
                mask,
                flat_latlons,
                months,
                y,
                x_coord,
                y_coord,
            ) = ds[i]
            dl = DataLoader(
                TensorDataset(
                    torch.from_numpy(eo).float(),
                    torch.from_numpy(y.astype(np.int16)),
                    torch.from_numpy(dynamic_world).long(),
                    torch.from_numpy(flat_latlons).float(),
                    torch.from_numpy(months).long(),
                    torch.from_numpy(mask).float(),
                ),
                batch_size=512,
                shuffle=False,
            )
            test_preds_np, _ = self._inference_for_dl(dl, finetuned_model, pretrained_model)

            # take the middle timestep's ndvi
            middle_timestep = eo.shape[1] // 2
            ndvi = eo[:, middle_timestep, NORMED_BANDS.index("NDVI")]
            da = ds.combine_predictions(test_preds_np, y, ndvi, x_coord, y_coord)
            prefix = f"{self.name}_{ds.all_files[i].stem}"
            if pretrained_model is None:
                filename = f"{prefix}_finetuning.nc"
            else:
                filename = f"{prefix}_{finetuned_model.__class__.__name__}.nc"
            da.to_netcdf(self.spatial_inference_savedir / filename)

    @torch.no_grad()
    def evaluate(
        self,
        finetuned_model: Union[PrestoFineTuningModel, BaseEstimator],
        pretrained_model: Optional[PrestoFineTuningModel] = None,
    ) -> Dict:

        test_ds = self.ds_class(self.test_df, target_function=self.target_function)
        dl = DataLoader(
            test_ds,
            batch_size=512,
            shuffle=False,  # keep as False!
            num_workers=Hyperparams.num_workers,
        )
        assert isinstance(dl.sampler, torch.utils.data.SequentialSampler)

        test_preds_np, target_np = self._inference_for_dl(dl, finetuned_model, pretrained_model)
        test_preds_np = test_preds_np >= self.threshold
        prefix = f"{self.name}_{finetuned_model.__class__.__name__}"

        catboost_preds = test_ds.df.worldcereal_prediction

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
            **format_partitioned(self.partitioned_metrics(target_np, test_preds_np, test_ds.df)),
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
        self,
        target: np.ndarray,
        preds: np.ndarray,
        test_df: pd.DataFrame,
    ) -> Dict[str, Union[np.float32, np.int32]]:
        catboost_preds = test_df.worldcereal_prediction
        years = test_df.end_date.apply(lambda date: date[:4])

        if "continent" not in test_df.columns:
            # might not be None if we have filtered by country
            world_attrs = WorldCerealLabelledDataset.join_with_world_df(test_df)[
                ["aez_zoneid", "name", "continent", "region"]
            ]
        else:
            world_attrs = test_df[["aez_zoneid", "name", "continent", "region"]]

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

        train_ds = self.ds_class(
            self.train_df,
            countries_to_remove=self.countries_to_remove,
            years_to_remove=self.years_to_remove,
            target_function=self.target_function,
            balance=True,
            mask_ratio=self.train_masking,
        )

        # should the val set be balanced too?
        val_ds = self.ds_class(
            self.val_df,
            countries_to_remove=self.countries_to_remove,
            years_to_remove=self.years_to_remove,
            target_function=self.target_function,
            mask_ratio=0.0,  # https://github.com/WorldCereal/presto-worldcereal/pull/102
        )

        loss_fn = nn.BCEWithLogitsLoss()

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

        for _ in (pbar := tqdm(range(hyperparams.max_epochs), desc="Finetuning")):
            model.train()
            epoch_train_loss = 0.0
            for x, y, dw, latlons, month, variable_mask in tqdm(
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
            for x, y, dw, latlons, month, variable_mask in val_dl:
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
            pbar.set_description(f"Train metric: {train_loss[-1]}, Val metric: {val_loss[-1]}")

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

    def finetuning_results_sklearn(
        self, sklearn_model_modes: List[str], finetuned_model: PrestoFineTuningModel
    ) -> Dict:
        results_dict = {}
        if len(sklearn_model_modes) > 0:
            dl = DataLoader(
                self.ds_class(
                    self.train_df,
                    countries_to_remove=self.countries_to_remove,
                    years_to_remove=self.years_to_remove,
                    target_function=self.target_function,
                    mask_ratio=self.train_masking,
                ),
                batch_size=2048,
                shuffle=False,
                num_workers=4,
            )
            val_dl = DataLoader(
                self.ds_class(
                    self.val_df,
                    countries_to_remove=self.countries_to_remove,
                    years_to_remove=self.years_to_remove,
                    target_function=self.target_function,
                    mask_ratio=0.0,  # https://github.com/WorldCereal/presto-worldcereal/pull/102
                ),
                batch_size=2048,
                shuffle=False,
                num_workers=4,
            )
            sklearn_models = self.finetune_sklearn_model(
                dl,
                val_dl,
                finetuned_model,
                models=sklearn_model_modes,
            )
            for sklearn_model in sklearn_models:
                logger.info(f"Evaluating {sklearn_model}...")
                results_dict.update(self.evaluate(sklearn_model, finetuned_model))
                if self.spatial_inference_savedir is not None:
                    self.spatial_inference(sklearn_model, finetuned_model)
        return results_dict

    def finetuning_results(
        self,
        pretrained_model,
        sklearn_model_modes: List[str],
    ) -> Tuple[Dict, PrestoFineTuningModel]:
        for model_mode in sklearn_model_modes:
            assert model_mode in ["Regression", "Random Forest", "CatBoostClassifier"]

        results_dict = {}
        # we want to always finetune the model, since the sklearn models
        # will use the finetuned model as a base. This better reflects
        # the deployment scenario for WorldCereal
        finetuned_model = self.finetune(pretrained_model)
        results_dict.update(self.evaluate(finetuned_model, None))
        if self.spatial_inference_savedir is not None:
            self.spatial_inference(finetuned_model, None)
        results_dict.update(self.finetuning_results_sklearn(sklearn_model_modes, finetuned_model))
        return results_dict, finetuned_model
