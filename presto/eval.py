import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from hiclass import LocalClassifierPerNode
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .dataset import (
    NORMED_BANDS,
    WorldCerealInferenceDataset,
    WorldCerealLabelled10DDataset,
    WorldCerealLabelledDataset,
)
from .hierarchical_classification import CatBoostClassifierWrapper
from .presto import (
    Presto,
    PrestoFineTuningModel,
    get_sinusoid_encoding_table,
    param_groups_lrd,
)
from .utils import DEFAULT_SEED, device, get_class_mappings, prep_dataframe

MIN_SAMPLES_PER_CLASS = 3

logger = logging.getLogger("__main__")

SklearnStyleModel = Union[BaseEstimator, CatBoostClassifier]


@dataclass
class Hyperparams:
    lr: float = 2e-5
    max_epochs: int = 100
    batch_size: int = 256
    patience: int = 20
    num_workers: int = 8
    catboost_iterations: int = 8000


class WorldCerealEval:
    threshold = 0.5
    regression = False

    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        countries_to_remove: Optional[List[str]] = None,
        years_to_remove: Optional[List[int]] = None,
        spatial_inference_savedir: Optional[Path] = None,
        seed: int = DEFAULT_SEED,
        filter_function: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        name: Optional[str] = None,
        val_size: float = 0.2,
        dekadal: bool = False,
        task_type: str = "cropland",
        num_outputs: int = 1,
        croptype_list: List = [],
        finetune_classes: str = "CROPTYPE0",
        downstream_classes: str = "CROPTYPE9",
        balance: bool = False,
        augment: bool = False,
        train_masking: float = 0.0,
        use_valid_month: bool = True,
    ):
        self.seed = seed
        self.task_type = task_type
        self.name = f"WorldCereal{task_type.title()}"

        train_data, val_data = WorldCerealLabelledDataset.split_df(train_data, val_size=val_size)

        self.train_df = prep_dataframe(train_data, filter_function, dekadal=dekadal)
        self.val_df = prep_dataframe(val_data, filter_function, dekadal=dekadal)
        self.test_df = prep_dataframe(test_data, filter_function, dekadal=dekadal)

        if task_type == "cropland":
            self.num_outputs = 1
            self.croptype_list = []
        elif task_type == "croptype":
            # compress all classes in train that contain less
            # than MIN_SAMPLES_PER_CLASS samples into "other"
            for class_column in ["finetune_class", "downstream_class"]:
                class_counts = self.train_df[class_column].value_counts()
                small_classes = class_counts[class_counts < MIN_SAMPLES_PER_CLASS].index
                # if no classes with n_samples < classes_threshold are present in train,
                # force the "other" class using the class with minimal number of samples
                # this is done so that the other class is always present,
                # thus making test set with new labels compatible with the model,
                # as in this way unseen labels will be mapped into "other" class
                if len(small_classes) == 0:
                    small_classes = [class_counts.index[-1]]

                self.train_df.loc[
                    self.train_df[class_column].isin(small_classes), class_column
                ] = "other_crop"
                train_classes = list(self.train_df[class_column].unique())
                if class_column == "finetune_class":
                    self.croptype_list = train_classes
                    self.num_outputs = len(train_classes)

                # use classes obtained from train to trim val and test classes
                self.val_df.loc[
                    ~self.val_df[class_column].isin(train_classes), class_column
                ] = "other_crop"
                self.test_df.loc[
                    ~self.test_df[class_column].isin(train_classes), class_column
                ] = "other_crop"

            # create one-hot representation from obtained labels
            # one-hot is needed for finetuning,
            # while downstream CatBoost can work with categorical labels
            self.train_df["finetune_class_oh"] = self.train_df["finetune_class"].copy()
            self.train_df = pd.get_dummies(
                self.train_df, prefix="", prefix_sep="", columns=["finetune_class_oh"]
            )
            # for test and val, additional step is needed to check
            # whether certain classes are missing and need to be forced into df
            self.val_df = self.convert_to_onehot(self.val_df, self.croptype_list)
            self.test_df = self.convert_to_onehot(self.test_df, self.croptype_list)

            self.finetune_classes = finetune_classes
            self.downstream_classes = downstream_classes
        else:
            logger.error(
                "Unknown task type. \
                Make sure that task type is one on the following: [cropland, croptype]"
            )

        self.spatial_inference_savedir = spatial_inference_savedir

        self.countries_to_remove = countries_to_remove
        self.years_to_remove = years_to_remove

        if self.countries_to_remove is not None:
            self.name = f"{self.name}_removed_countries_{countries_to_remove}"
        if self.years_to_remove is not None:
            self.name = f"{self.name}_removed_years_{years_to_remove}"

        self.dekadal = dekadal
        self.balance = balance
        self.ds_class = WorldCerealLabelled10DDataset if dekadal else WorldCerealLabelledDataset
        self.train_masking = train_masking
        self.augment = augment
        self.use_valid_month = use_valid_month
        logger.info(f"Usage of time token is {'enabled' if use_valid_month else 'disabled'}.")
        logger.info(
            f"Mask ratio is {'enabled' if train_masking>0 else 'disabled'}, {train_masking}."
        )

    @staticmethod
    def convert_to_onehot(
        df: pd.DataFrame,
        croptype_list: List = [],
    ):
        df["finetune_class_oh"] = df["finetune_class"].copy()
        df = pd.get_dummies(df, prefix="", prefix_sep="", columns=["finetune_class_oh"])
        cols_to_add = [xx for xx in croptype_list if xx not in df.columns]
        if len(cols_to_add) > 0:
            for col in cols_to_add:
                df[col] = 0

        return df

    def _construct_finetuning_model(self, pretrained_model: Presto) -> PrestoFineTuningModel:
        model: PrestoFineTuningModel = cast(Callable, pretrained_model.construct_finetuning_model)(
            num_outputs=self.num_outputs
        )
        if self.dekadal:
            max_sequence_length = 72  # can this be 36?
            model.encoder.pos_embed = nn.Parameter(
                torch.zeros(1, max_sequence_length, model.encoder.pos_embed.shape[-1]),
                requires_grad=False,
            )
            pos_embed = get_sinusoid_encoding_table(
                model.encoder.pos_embed.shape[1], model.encoder.pos_embed.shape[-1]
            )
            model.encoder.pos_embed.data.copy_(pos_embed)
        model.to(device)
        return model

    @torch.no_grad()
    def finetune_sklearn_model(
        self,
        pretrained_model: PrestoFineTuningModel,
        models: List[str] = ["Regression", "Random Forest"],
        hyperparams: Hyperparams = Hyperparams(),
    ) -> Union[Sequence[BaseEstimator], Dict]:
        for model_mode in models:
            assert model_mode in [
                "Regression",
                "Random Forest",
                "CatBoostClassifier",
                "Hierarchical CatBoostClassifier",
            ]
        pretrained_model.eval()

        def dataloader_to_encodings_and_targets(
            dl: DataLoader,
        ) -> Tuple[np.ndarray, np.ndarray]:
            encoding_list, target_list = [], []
            for x, y, dw, latlons, month, valid_month, variable_mask in dl:
                x_f, dw_f, latlons_f, month_f, valid_month_f, variable_mask_f = [
                    t.to(device) for t in (x, dw, latlons, month, valid_month, variable_mask)
                ]
                if isinstance(y, list) and len(y) == 2:
                    y = np.moveaxis(np.array(y), -1, 0)
                target_list.append(y)

                if self.use_valid_month:
                    with torch.no_grad():
                        encodings = (
                            pretrained_model.encoder(
                                x_f,
                                dynamic_world=dw_f.long(),
                                mask=variable_mask_f,
                                latlons=latlons_f,
                                month=month_f,
                                valid_month=valid_month_f,
                            )
                            .cpu()
                            .numpy()
                        )
                        encoding_list.append(encodings)
                else:
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

        if self.task_type == "cropland":
            eval_metric = "F1"
            loss_function = "Logloss"
        if self.task_type == "croptype":
            eval_metric = "MultiClass"
            loss_function = "MultiClass"

        fit_models = []
        for model in tqdm(models, desc="Fitting sklearn models"):
            dl = DataLoader(
                self.ds_class(
                    self.train_df,
                    countries_to_remove=self.countries_to_remove,
                    years_to_remove=self.years_to_remove,
                    task_type=self.task_type,
                    croptype_list=[],
                    return_hierarchical_labels="Hierarchical" in model,
                ),
                batch_size=2048,
                shuffle=False,
                num_workers=8,
            )
            val_dl = DataLoader(
                self.ds_class(
                    self.val_df,
                    countries_to_remove=self.countries_to_remove,
                    years_to_remove=self.years_to_remove,
                    task_type=self.task_type,
                    croptype_list=[],
                    return_hierarchical_labels="Hierarchical" in model,
                ),
                batch_size=2048,
                shuffle=False,
                num_workers=8,
            )

            train_encodings, train_targets = dataloader_to_encodings_and_targets(dl)
            val_encodings, val_targets = dataloader_to_encodings_and_targets(val_dl)

            if model != "Hierarchical CatBoostClassifier":
                class_weights = cast(WorldCerealLabelledDataset, dl.dataset).class_weights
                class_weight_dict = dict(zip(np.unique(train_targets), class_weights))
                sample_weights_trn = np.ones((len(train_targets),))
                sample_weights_val = np.ones((len(val_targets),))
                if self.balance:
                    for k, v in class_weight_dict.items():
                        sample_weights_trn[train_targets == k] = v
                        sample_weights_val[val_targets == k] = v

            if model == "CatBoostClassifier":
                # Parameters emulate
                # # https://github.com/WorldCereal/wc-classification/blob/
                # # 4a9a839507d9b4f63c378b3b1d164325cbe843d6/src/
                # # worldcereal/classification/models.py#L490

                if self.task_type == "cropland":
                    learning_rate = 0.05
                    l2_leaf_reg = 3
                if self.task_type == "croptype":
                    learning_rate = 0.1
                    l2_leaf_reg = 30

                downstream_model = CatBoostClassifier(
                    iterations=hyperparams.catboost_iterations,
                    depth=8,
                    learning_rate=learning_rate,
                    early_stopping_rounds=50,
                    l2_leaf_reg=l2_leaf_reg,
                    eval_metric=eval_metric,
                    loss_function=loss_function,
                    random_state=self.seed,
                    verbose=100,
                    class_names=np.unique(train_targets),
                )

                fit_models.append(
                    clone(downstream_model).fit(
                        train_encodings,
                        train_targets,
                        sample_weight=sample_weights_trn,
                        eval_set=Pool(val_encodings, val_targets, weight=sample_weights_val),
                    )
                )

            elif model == "Hierarchical CatBoostClassifier":
                downstream_model = CatBoostClassifierWrapper(
                    iterations=hyperparams.catboost_iterations,
                    depth=8,
                    eval_metric="F1",
                    learning_rate=0.2,
                    l2_leaf_reg=100,
                    verbose=100,
                    random_seed=self.seed,
                )
                fit_models.append(
                    LocalClassifierPerNode(
                        local_classifier=downstream_model,
                        binary_policy="exclusive_siblings",
                    ).fit(
                        np.concatenate((train_encodings, val_encodings), axis=0),
                        np.concatenate((train_targets, val_targets), axis=0),
                    )
                )
            elif model == "Regression":
                downstream_model = LogisticRegression(
                    class_weight=class_weight_dict,
                    max_iter=1000,
                    random_state=self.seed,
                )
                fit_models.append(clone(downstream_model).fit(train_encodings, train_targets))
            elif model == "Random Forest":
                downstream_model = RandomForestClassifier(
                    class_weight=class_weight_dict,
                    random_state=self.seed,
                )
                fit_models.append(clone(downstream_model).fit(train_encodings, train_targets))
        return fit_models

    @staticmethod
    def _inference_for_dl(
        dl,
        finetuned_model: Union[PrestoFineTuningModel, SklearnStyleModel],
        pretrained_model: Optional[PrestoFineTuningModel] = None,
        task_type: str = "cropland",
        use_valid_month: bool = True,
    ) -> Tuple:
        test_preds, test_probs, targets = [], [], []
        for b in dl:
            x, y, dw, latlons, month, valid_month, variable_mask = b
            x_f, dw_f, latlons_f, month_f, valid_month_f, variable_mask_f = [
                t.to(device) for t in (x, dw, latlons, month, valid_month, variable_mask)
            ]
            if use_valid_month:
                input_d = {
                    "x": x_f,
                    "dynamic_world": dw_f.long(),
                    "latlons": latlons_f,
                    "mask": variable_mask_f,
                    "month": month_f,
                    "valid_month": valid_month_f,
                }
            else:
                input_d = {
                    "x": x_f,
                    "dynamic_world": dw_f.long(),
                    "latlons": latlons_f,
                    "mask": variable_mask_f,
                    "month": month_f,
                }

            # try:
            #     x, y, dw, latlons, month, valid_month, variable_mask = b
            #     x_f, dw_f, latlons_f, month_f, valid_month_f, variable_mask_f = [
            #         t.to(device) for t in (x, dw, latlons, month, valid_month, variable_mask)
            #     ]
            #     input_d = {
            #         "x": x_f,
            #         "dynamic_world": dw_f.long(),
            #         "latlons": latlons_f,
            #         "mask": variable_mask_f,
            #         "month": month_f,
            #         "valid_month": valid_month_f,
            #     }
            # except ValueError:
            #     x, y, dw, latlons, month, variable_mask = b
            #     x_f, dw_f, latlons_f, month_f, variable_mask_f = [
            #         t.to(device) for t in (x, dw, latlons, month, variable_mask)
            #     ]
            #     input_d = {
            #         "x": x_f,
            #         "dynamic_world": dw_f.long(),
            #         "latlons": latlons_f,
            #         "mask": variable_mask_f,
            #         "month": month_f,
            #     }

            if isinstance(y, list) and len(y) == 2:
                y = np.moveaxis(np.array(y), -1, 0)
            targets.append(y)
            if isinstance(finetuned_model, PrestoFineTuningModel) or isinstance(
                finetuned_model, Presto
            ):
                finetuned_model.eval()
                preds = finetuned_model(**input_d).squeeze(dim=1)
                if task_type == "cropland":
                    preds = torch.sigmoid(preds).cpu().numpy()
                    probs = preds.copy()
                elif task_type == "croptype":
                    preds = nn.functional.softmax(preds, dim=1).cpu().numpy()
                    probs = preds.copy()
                else:
                    logger.error(
                        "Unknown task type. \
                            Make sure that task type is one on the following: [cropland, croptype]"
                    )
            else:
                cast(Presto, pretrained_model).eval()
                encodings = cast(Presto, pretrained_model).encoder(**input_d).cpu().numpy()
                if task_type == "cropland":
                    preds = finetuned_model.predict_proba(encodings)[:, 1]
                    probs = preds.copy()
                elif task_type == "croptype":
                    preds = finetuned_model.predict(encodings)
                    # for hierarchical classification, get predictions on the most granular level
                    if preds.ndim > 2:
                        preds = preds[:, -1]
                        probs = np.zeros_like(preds)
                    else:
                        probs = finetuned_model.predict_proba(encodings)
                else:
                    logger.error(
                        "Unknown task type. \
                            Make sure that task type is one on the following: [cropland, croptype]"
                    )

            test_preds.append(preds)
            test_probs.append(probs)

        test_preds_np = np.concatenate(test_preds)
        test_probs_np = np.concatenate(test_probs)
        target_np = np.concatenate(targets)

        return test_preds_np, test_probs_np, target_np

    @torch.no_grad()
    def spatial_inference(
        self,
        finetuned_model: Union[PrestoFineTuningModel, SklearnStyleModel],
        pretrained_model: Optional[PrestoFineTuningModel] = None,
    ):
        assert self.spatial_inference_savedir is not None

        CLASS_MAPPINGS = get_class_mappings()

        ds = WorldCerealInferenceDataset()
        for i in range(len(ds)):
            (
                eo,
                dynamic_world,
                mask,
                flat_latlons,
                months,
                y,
                valid_months,
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
                    torch.from_numpy(valid_months).long(),
                    torch.from_numpy(mask).float(),
                ),
                batch_size=2048,
                shuffle=False,
            )
            test_preds_np, test_probs_np, _ = self._inference_for_dl(
                dl,
                finetuned_model,
                pretrained_model,
                task_type=self.task_type,
                use_valid_month=self.use_valid_month,
            )
            test_preds_str = test_preds_np.copy()

            if self.task_type == "croptype":
                if pretrained_model is None:
                    temp_croptype_map = pd.DataFrame(
                        CLASS_MAPPINGS[self.finetune_classes].items(),
                        columns=["ewoc_code", "name"],
                    )
                    test_preds_np = np.argmax(test_preds_np, axis=-1)
                    test_preds_str = np.array([self.croptype_list[xx] for xx in test_preds_np])
                else:
                    test_preds_np = np.argmax(test_probs_np, axis=-1)
                    temp_croptype_map = pd.DataFrame(
                        CLASS_MAPPINGS[self.downstream_classes].items(),
                        columns=["ewoc_code", "name"],
                    )

                temp_croptype_map.sort_values(
                    by=["name", "ewoc_code"], ascending=True, inplace=True
                )
                temp_croptype_map.drop_duplicates(subset=["name"], keep="first", inplace=True)
                temp_croptype_map.set_index("name", inplace=True)

                if test_preds_str.ndim > 1:
                    test_preds_str = test_preds_str.flatten()

                test_preds_ewoc_code = np.array(
                    [int(temp_croptype_map.loc[xx].iloc[0]) for xx in test_preds_str]
                )

            # take the middle timestep's ndvi
            middle_timestep = eo.shape[1] // 2
            ndvi = eo[:, middle_timestep, NORMED_BANDS.index("NDVI")]

            b2 = eo[:, middle_timestep, NORMED_BANDS.index("B2")]
            b3 = eo[:, middle_timestep, NORMED_BANDS.index("B3")]
            b4 = eo[:, middle_timestep, NORMED_BANDS.index("B4")]

            if self.task_type == "cropland":
                da = ds.combine_predictions(
                    x_coord,
                    y_coord,
                    test_preds_np,
                    test_preds_np,
                    test_preds_np,
                    y,
                    ndvi,
                    b2,
                    b3,
                    b4,
                )
            if self.task_type == "croptype":
                da = ds.combine_predictions(
                    x_coord,
                    y_coord,
                    test_preds_np,
                    test_preds_ewoc_code,
                    test_probs_np,
                    y,
                    ndvi,
                    b2,
                    b3,
                    b4,
                )

            prefix = f"{self.name}_{ds.all_files[i].stem}"
            if pretrained_model is None:
                filename = f"{prefix}_finetuning_{self.task_type}.nc"
            else:
                filename = f"{prefix}_{finetuned_model.__class__.__name__}_{self.task_type}.nc"
            da.to_netcdf(self.spatial_inference_savedir / filename)

    @torch.no_grad()
    def evaluate(
        self,
        finetuned_model: Union[PrestoFineTuningModel, BaseEstimator],
        pretrained_model: Optional[PrestoFineTuningModel] = None,
        croptype_list: List = [],
    ) -> pd.DataFrame:

        test_ds = self.ds_class(
            self.test_df,
            task_type=self.task_type,
            croptype_list=croptype_list,
            return_hierarchical_labels=(
                type(finetuned_model).__name__
                in ["LocalClassifierPerParentNode", "LocalClassifierPerNode"]
            ),
        )

        dl = DataLoader(
            test_ds,
            batch_size=2048,
            shuffle=False,  # keep as False!
            num_workers=Hyperparams.num_workers,
        )
        assert isinstance(dl.sampler, torch.utils.data.SequentialSampler)

        test_preds_np, test_probs_np, target_np = self._inference_for_dl(
            dl,
            finetuned_model,
            pretrained_model,
            task_type=self.task_type,
            use_valid_month=self.use_valid_month,
        )
        if self.task_type == "cropland":
            test_preds_np = test_preds_np >= self.threshold
            _croptype_list = ["not_crop", "crop"]
        if self.task_type == "croptype":
            if len(croptype_list) > 0:
                test_preds_np = np.argmax(test_preds_np, axis=-1)
                test_preds_np = np.array([self.croptype_list[xx] for xx in test_preds_np])

                target_np = np.argmax(target_np, axis=-1)
                target_np = np.array([self.croptype_list[xx] for xx in target_np])
            elif target_np.ndim > 1:
                # for hierarchical classification, get predictions on the most granular level
                target_np = np.array(target_np)[:, -1]

            _croptype_list = list(np.unique(target_np))

        if self.task_type == "cropland":
            metrics_agg = "binary"
            target_np = np.array(["crop" if xx > 0.5 else "not_crop" for xx in target_np])
            test_preds_np = np.array(["crop" if xx > 0.5 else "not_crop" for xx in test_preds_np])
        if self.task_type == "croptype":
            metrics_agg = "macro"

        _results = classification_report(
            target_np,
            test_preds_np,
            labels=_croptype_list,
            output_dict=True,
            # zero_division=np.nan,
            zero_division=0,
        )

        _results_df = pd.DataFrame(_results).transpose().reset_index()
        _results_df.columns = pd.Index(["class", "precision", "recall", "f1-score", "support"])
        _results_df["year"] = "all"
        _results_df["country"] = "all"

        # overwrite macro F1 so that it's not computed for classes that have
        # too few samples
        corrected_macro_f1 = _results_df.loc[
            (_results_df["support"] >= MIN_SAMPLES_PER_CLASS)
            & (~_results_df["class"].isin(["accuracy", "macro avg", "weighted avg"])),
            "f1-score",
        ].mean()
        _results_df.loc[_results_df["class"] == "macro avg", "f1-score"] = corrected_macro_f1
        _results_df.loc[_results_df["class"] == "macro avg", "f1-score"] = (
            corrected_macro_f1 if not np.isnan(corrected_macro_f1) else 0
        )

        _partitioned_results = self.partitioned_metrics(
            target_np, test_preds_np, test_ds.df, metrics_agg, _croptype_list
        )

        _results_df = pd.concat((_results_df, _partitioned_results), axis=0)
        _results_df["downstream_model_type"] = type(finetuned_model).__name__

        return _results_df

    def partitioned_metrics(
        self,
        target: np.ndarray,
        preds: np.ndarray,
        test_df: pd.DataFrame,
        metrics_agg: str,
        croptype_list: List = [],
    ) -> pd.DataFrame:
        partitioned_result_df = pd.DataFrame()
        test_df = WorldCerealLabelledDataset.join_with_world_df(test_df)
        for prop_name in ["year", "name"]:
            prop_series = test_df[prop_name]
            for prop in prop_series.dropna().unique():
                f: pd.Series = cast(pd.Series, prop_series == prop)
                _report = classification_report(
                    target[f],
                    preds[f],
                    labels=croptype_list,
                    output_dict=True,
                    zero_division=0,
                )
                _report_df = pd.DataFrame(_report).transpose().reset_index()
                _report_df.columns = pd.Index(
                    [
                        "class",
                        "precision",
                        "recall",
                        "f1-score",
                        "support",
                    ]
                )
                if prop_name == "year":
                    _report_df["year"] = prop
                    _report_df["country"] = "all"
                if prop_name == "name":
                    _report_df["year"] = "all"
                    _report_df["country"] = prop

                corrected_macro_f1 = _report_df.loc[
                    (_report_df["support"] >= MIN_SAMPLES_PER_CLASS)
                    & (~_report_df["class"].isin(["accuracy", "macro avg", "weighted avg"])),
                    "f1-score",
                ].mean()
                _report_df.loc[_report_df["class"] == "macro avg", "f1-score"] = (
                    corrected_macro_f1 if not np.isnan(corrected_macro_f1) else 0
                )

                partitioned_result_df = pd.concat([partitioned_result_df, _report_df], axis=0)

        return partitioned_result_df

    def finetune(
        self, pretrained_model, hyperparams: Hyperparams = Hyperparams()
    ) -> PrestoFineTuningModel:
        model = self._construct_finetuning_model(pretrained_model)

        parameters = param_groups_lrd(model)

        if self.task_type == "croptype":
            hyperparams.lr = 0.01

        optimizer = AdamW(parameters, lr=hyperparams.lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        train_ds = self.ds_class(
            self.train_df,
            balance=self.balance,
            task_type=self.task_type,
            croptype_list=self.croptype_list,
            augment=self.augment,
            mask_ratio=self.train_masking,
        )

        # should the val set be balanced too?
        val_ds = self.ds_class(
            self.val_df,
            countries_to_remove=self.countries_to_remove,
            years_to_remove=self.years_to_remove,
            augment=False,  # don't augment the validation set
            task_type=self.task_type,
            croptype_list=self.croptype_list,
            mask_ratio=0.0,  # https://github.com/WorldCereal/presto-worldcereal/pull/102
        )

        loss_fn: nn.Module
        if self.task_type == "croptype":
            loss_fn = nn.CrossEntropyLoss()
        if self.task_type == "cropland":
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
            for x, y, dw, latlons, month, valid_month, variable_mask in tqdm(
                train_dl, desc="Training", leave=False
            ):
                x, y, dw, latlons, month, valid_month, variable_mask = [
                    t.to(device) for t in (x, y, dw, latlons, month, valid_month, variable_mask)
                ]

                if self.use_valid_month:
                    input_d = {
                        "x": x,
                        "dynamic_world": dw.long(),
                        "latlons": latlons,
                        "mask": variable_mask,
                        "month": month,
                        "valid_month": valid_month,
                    }
                else:
                    input_d = {
                        "x": x,
                        "dynamic_world": dw.long(),
                        "latlons": latlons,
                        "mask": variable_mask,
                        "month": month,
                    }

                optimizer.zero_grad()
                preds = model(**input_d)
                loss = loss_fn(preds.squeeze(-1), y.float())
                epoch_train_loss += loss.item()
                loss.backward()
                optimizer.step()
            scheduler.step()
            train_loss.append(epoch_train_loss / len(train_dl))

            model.eval()
            all_preds, all_y = [], []
            for x, y, dw, latlons, month, valid_month, variable_mask in val_dl:
                x, y, dw, latlons, month, valid_month, variable_mask = [
                    t.to(device) for t in (x, y, dw, latlons, month, valid_month, variable_mask)
                ]

                if self.use_valid_month:
                    input_d = {
                        "x": x,
                        "dynamic_world": dw.long(),
                        "latlons": latlons,
                        "mask": variable_mask,
                        "month": month,
                        "valid_month": valid_month,
                    }
                else:
                    input_d = {
                        "x": x,
                        "dynamic_world": dw.long(),
                        "latlons": latlons,
                        "mask": variable_mask,
                        "month": month,
                    }

                with torch.no_grad():
                    preds = model(**input_d)
                    all_preds.append(preds.squeeze(-1))
                    all_y.append(y.float())

            val_loss.append(loss_fn(torch.cat(all_preds), torch.cat(all_y)))

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

            pbar.set_description(
                f"Train metric: {train_loss[-1]:.3f}, Val metric: {val_loss[-1]:.3f}, \
                Best Val Loss: {best_loss:.3f} \
                (no improvement for {epochs_since_improvement} epochs)"
            )
            if run is not None:
                wandb.log(
                    {
                        f"{self.name}_finetuning_val_loss": val_loss[-1],
                        f"{self.name}_finetuning_train_loss": train_loss[-1],
                    }
                )

        assert best_model_dict is not None
        model.load_state_dict(best_model_dict)

        model.eval()
        return model

    def finetuning_results_sklearn(
        self,
        sklearn_model_modes: List[str],
        finetuned_model: PrestoFineTuningModel,
        hyperparams: Hyperparams = Hyperparams(),
    ):
        results_df = pd.DataFrame()
        if len(sklearn_model_modes) > 0:
            sklearn_models = self.finetune_sklearn_model(
                finetuned_model,
                models=sklearn_model_modes,
                hyperparams=hyperparams
            )
            for sklearn_model in sklearn_models:
                logger.info(f"Evaluating {type(sklearn_model).__name__}...")
                _results_df = self.evaluate(sklearn_model, finetuned_model, croptype_list=[])
                results_df = pd.concat([results_df, _results_df], axis=0)
                if self.spatial_inference_savedir is not None:
                    self.spatial_inference(sklearn_model, finetuned_model)
        return results_df, sklearn_models

    def finetuning_results(
        self,
        pretrained_model,
        sklearn_model_modes: List[str],
        hyperparams: Hyperparams = Hyperparams(),
    ) -> Tuple[pd.DataFrame, PrestoFineTuningModel, List]:
        for model_mode in sklearn_model_modes:
            assert model_mode in [
                "Regression",
                "Random Forest",
                "CatBoostClassifier",
                "Hierarchical CatBoostClassifier",
            ]

        finetuned_model = self.finetune(pretrained_model, hyperparams=hyperparams)
        print("Finetuning done")
        results_df_ft = self.evaluate(finetuned_model, None, croptype_list=self.croptype_list)
        print("Finetuning head evaluation done")
        if self.spatial_inference_savedir is not None:
            self.spatial_inference(finetuned_model, None)
        results_df_sklearn, sklearn_models_trained = self.finetuning_results_sklearn(
            sklearn_model_modes, finetuned_model, hyperparams
        )
        results_df_combined = pd.concat([results_df_ft, results_df_sklearn], axis=0)

        return results_df_combined, finetuned_model, sklearn_models_trained
