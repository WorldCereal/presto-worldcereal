import logging
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from einops import repeat
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import utils
from .dataset import WorldCerealLabelledDataset
from .presto import Presto, PrestoFineTuningModel
from .utils import DEFAULT_SEED, device

logger = logging.getLogger("__main__")

world_shp_path = "world_shp/world-administrative-boundaries.shp"


class WorldCerealEval:
    name = "WorldCerealCropland"
    threshold = 0.5

    def __init__(self, train_data: pd.DataFrame, val_data: pd.DataFrame, seed: int = DEFAULT_SEED):
        self.seed = seed
        self.train_df = train_data
        self.val_df = val_data

        # todo temporary
        new_val = pd.read_parquet(
            utils.data_dir / "worldcereal_presto_cropland_linearinterp_V2_VAL.parquet"
        )
        new_val = new_val.drop_duplicates(subset=["lat", "lon", "end_date", "start_date"])
        self.val_df = pd.merge(
            self.val_df,
            new_val,
            how="left",
            on=["lat", "lon", "end_date", "start_date"],
            suffixes=(None, "_new"),
            validate="many_to_one",
        )

        self.val_df = self.val_df[~pd.isna(self.val_df).any(axis=1)]

        self.world_shp = gpd.read_file(utils.data_dir / world_shp_path)
        # these columns contain nan sometimes causing difficulties
        self.world_shp = self.world_shp.drop(
            columns=["iso3", "status", "color_code", "iso_3166_1_"]
        )

    @staticmethod
    def _mask_to_batch_tensor(
        mask: Optional[np.ndarray], batch_size: int
    ) -> Optional[torch.Tensor]:
        # TODO: This function should be replaced by a real mask,
        # returned by the dataloader
        if mask is not None:
            return repeat(torch.from_numpy(mask).to(device), "t c -> b t c", b=batch_size).float()
        return None

    @torch.no_grad()
    def finetune_sklearn_model(
        self,
        dl: DataLoader,
        pretrained_model,
        mask: Optional[np.ndarray] = None,
        models: List[str] = ["Regression", "Random Forest"],
    ) -> Union[Sequence[BaseEstimator], Dict]:
        for model_mode in models:
            assert model_mode in ["Regression", "Random Forest"]
        pretrained_model.eval()

        encoding_list, target_list = [], []
        for x, y, dw, latlons, month in tqdm(dl, desc="Computing embeddings"):
            x, dw, latlons, y, month = [t.to(device) for t in (x, dw, latlons, y, month)]
            batch_mask = self._mask_to_batch_tensor(mask, x.shape[0])
            target_list.append(y.cpu().numpy())
            with torch.no_grad():
                encodings = (
                    pretrained_model.encoder(
                        x, dynamic_world=dw.long(), mask=batch_mask, latlons=latlons, month=month
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
        pretrained_model=None,
        mask: Optional[np.ndarray] = None,
    ) -> Dict:

        if isinstance(finetuned_model, BaseEstimator):
            assert isinstance(pretrained_model, Presto)

        val_ds = WorldCerealLabelledDataset(self.val_df)
        dl = DataLoader(
            val_ds,
            batch_size=8192,
            shuffle=False,  # keep as False!
            num_workers=8,
        )
        assert isinstance(dl.sampler, torch.utils.data.SequentialSampler)

        test_preds, targets = [], []
        for x, y, dw, latlons, month in dl:
            targets.append(y.cpu().numpy())
            x, dw, latlons, month = [t.to(device) for t in (x, dw, latlons, month)]
            batch_mask = self._mask_to_batch_tensor(mask, x.shape[0])
            if isinstance(finetuned_model, PrestoFineTuningModel):
                finetuned_model.eval()
                preds = (
                    finetuned_model(
                        x, dynamic_world=dw.long(), mask=batch_mask, latlons=latlons, month=month
                    )
                    .squeeze(dim=1)
                    .cpu()
                    .numpy()
                )
            elif isinstance(finetuned_model, BaseEstimator):
                cast(Presto, pretrained_model).eval()
                encodings = (
                    cast(Presto, pretrained_model)
                    .encoder(
                        x, dynamic_world=dw.long(), mask=batch_mask, latlons=latlons, month=month
                    )
                    .cpu()
                    .numpy()
                )
                preds = finetuned_model.predict(encodings)
            test_preds.append(preds)

        test_preds_np = np.concatenate(test_preds) >= self.threshold
        target_np = np.concatenate(targets)
        prefix = f"{self.name}_{finetuned_model.__class__.__name__}"

        val_df = self.val_df.loc[
            ~self.val_df.LANDCOVER_LABEL.isin(WorldCerealLabelledDataset.FILTER_LABELS)
        ]
        catboost_preds = val_df.catboost_prediction

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
            recalls.append(recall_score(target[f], preds[f], zero_division=np.nan))
            precisions.append(precision_score(target[f], preds[f], zero_division=np.nan))
            res.update(
                {
                    f"{prefix}_num_samples: {prop}": f.sum(),
                    f"{prefix}_f1: {prop}": f1_score(target[f], preds[f], zero_division=np.nan),
                    f"{prefix}_recall: {prop}": recalls[-1],
                    f"{prefix}_precision: {prop}": precisions[-1],
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

        val_df = self.val_df.loc[
            ~self.val_df.LANDCOVER_LABEL.isin(WorldCerealLabelledDataset.FILTER_LABELS)
        ]
        catboost_preds = val_df.catboost_prediction
        years = val_df.end_date.apply(lambda date: date[:4])

        latlons = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries.from_xy(x=val_df.lon, y=val_df.lat), crs="EPSG:4326"
        )
        # project to non geographic CRS, otherwise geopandas gives a warning
        world_attrs = gpd.sjoin_nearest(
            latlons.to_crs("EPSG:3857"), self.world_shp.to_crs("EPSG:3857"), how="left"
        )
        world_attrs = world_attrs[~world_attrs.index.duplicated(keep="first")]
        if world_attrs.isna().any(axis=1).any():
            logger.warning("Some coordinates couldn't be matched to a country")

        metrics = partial(self.metrics, target=target)
        return {
            **metrics("aez", val_df.aez_zoneid, preds),
            **metrics("year", years, preds),
            **metrics("country", world_attrs.name, preds),
            **metrics("continent", world_attrs.continent, preds),
            **metrics("region", world_attrs.region, preds),
            **metrics("CatBoost_aez", val_df.aez_zoneid, catboost_preds),
            **metrics("CatBoost_year", years, catboost_preds),
            **metrics("CatBoost_country", world_attrs.name, catboost_preds),
            **metrics("CatBoost_continent", world_attrs.continent, catboost_preds),
            **metrics("CatBoost_region", world_attrs.region, catboost_preds),
        }

    def finetuning_results(
        self,
        pretrained_model,
        model_modes: List[str],
        mask: Optional[np.ndarray] = None,
    ) -> Dict:
        results_dict = {}
        for model_mode in model_modes:
            # TODO: "finetune"
            assert model_mode in ["Regression", "Random Forest"]

        sklearn_modes = [x for x in model_modes if x != "finetune"]
        if len(sklearn_modes) > 0:
            dl = DataLoader(
                WorldCerealLabelledDataset(self.train_df),
                batch_size=8192,
                shuffle=False,
                num_workers=8,
            )
            sklearn_models = self.finetune_sklearn_model(
                dl,
                pretrained_model,
                mask=mask,
                models=sklearn_modes,
            )
            for sklearn_model in sklearn_models:
                logger.info(f"Fitting {sklearn_model}...")
                results_dict.update(self.evaluate(sklearn_model, pretrained_model, mask))
        return results_dict
