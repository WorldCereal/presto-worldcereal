from typing import Dict, List, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import WorldCerealLabelledDataset
from .presto import Presto, PrestoFineTuningModel
from .utils import DEFAULT_SEED, device


class WorldCerealEval:
    name = "WorldCerealCropland"
    threshold = 0.5

    def __init__(self, train_data: pd.DataFrame, val_data: pd.DataFrame, seed: int = DEFAULT_SEED):
        self.seed = seed
        self.train_ds = train_data
        self.val_ds = val_data

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
        for x, y, dw, latlons, month, num_masked_tokens, variable_mask in dl:
            for masked_token_group in torch.unique(num_masked_tokens):
                filter = num_masked_tokens == masked_token_group
                target_list.append(y[filter].cpu().numpy())
                x_f, dw_f, latlons_f, month_f, variable_mask_f = [
                    t[filter].to(device) for t in (x, dw, latlons, month, variable_mask)
                ]
                target_list.append(y.cpu().numpy())
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
        pretrained_model=None,
        mask: Optional[np.ndarray] = None,
    ) -> Dict:

        if isinstance(finetuned_model, BaseEstimator):
            assert isinstance(pretrained_model, Presto)

        dl = DataLoader(
            WorldCerealLabelledDataset(self.val_ds),
            batch_size=8192,
            shuffle=False,
            num_workers=2,
        )

        test_preds, targets = [], []
        for x, y, dw, latlons, month, num_masked_tokens, variable_mask in dl:
            for masked_token_group in torch.unique(num_masked_tokens):
                filter = num_masked_tokens == masked_token_group
                targets.append(y[filter].cpu().numpy())
                x_f, dw_f, latlons_f, month_f, variable_mask_f = [
                    t[filter].to(device) for t in (x, dw, latlons, month, variable_mask)
                ]
                if isinstance(finetuned_model, PrestoFineTuningModel):
                    finetuned_model.eval()
                    preds = (
                        finetuned_model(
                            x_f,
                            dynamic_world=dw_f.long(),
                            mask=variable_mask_f,
                            latlons=latlons_f,
                            month=month_f,
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

        prefix = finetuned_model.__class__.__name__
        return {
            f"{self.name}_{prefix}_f1": f1_score(target_np, test_preds_np),
            f"{self.name}_{prefix}_recall": recall_score(target_np, test_preds_np),
            f"{self.name}_{prefix}_precision": precision_score(target_np, test_preds_np),
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
                WorldCerealLabelledDataset(self.train_ds),
                batch_size=8192,
                shuffle=False,
                num_workers=2,
            )
            sklearn_models = self.finetune_sklearn_model(
                dl,
                pretrained_model,
                mask=mask,
                models=sklearn_modes,
            )
            for sklearn_model in sklearn_models:
                results_dict.update(self.evaluate(sklearn_model, pretrained_model, mask))
        return results_dict
