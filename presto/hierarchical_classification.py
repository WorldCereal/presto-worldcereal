import logging

import numpy as np
from catboost import CatBoostClassifier
from hiclass import LocalClassifierPerNode, LocalClassifierPerParentNode
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array, check_is_fitted

logger = logging.getLogger("__main__")


class CatBoostClassifierWrapper(CatBoostClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y, *args, **kwargs):
        val_fraction = 0.3
        early_stopping_rounds = 100

        _X_trn, _X_val, _y_trn, _y_val = train_test_split(X, y, stratify=y, test_size=val_fraction)

        return super().fit(
            _X_trn,
            _y_trn,
            eval_set=(_X_val, _y_val),
            early_stopping_rounds=early_stopping_rounds,
        )


class LocalClassifierPerNodeWrapper(LocalClassifierPerNode):
    def __init__(
        self,
        local_classifier: None,
        binary_policy: str = "siblings",
        verbose: int = 0,
        edge_list: str = "",
        replace_classifiers: bool = True,
        n_jobs: int = 1,
        bert: bool = False,
    ):
        super().__init__(
            local_classifier=local_classifier,
            verbose=verbose,
            edge_list=edge_list,
            replace_classifiers=replace_classifiers,
        )
        self.binary_policy = binary_policy
        self.n_jobs = n_jobs
        self.bert = bert
        self.classifiers_ = {}  # type: ignore

    def _fit_local_classifier(self, node, X_node, y_node):
        nodes_to_train = [
            n for n in self.hierarchy_.nodes() if list(self.hierarchy_.successors(n))
        ]
        total = len(nodes_to_train)
        try:
            pos = nodes_to_train.index(node) + 1
        except ValueError:
            pos = "unknown"
        logger.info(f"Training classifier for class '{node}' ({pos} out of {total}).")
        clf = super()._fit_local_classifier(node, X_node, y_node)
        self.classifiers_[node] = clf
        return clf

    def predict_proba(self, X):
        """
        Predict probabilities level-by-level.
        For level 0 we use the classifiers associated with the children of the root.
        For subsequent levels, we use the classifier of the predicted parent.
        If a classifier is not found in self.classifiers_, we attempt to retrieve it from
        self.hierarchy_.nodes[node]["classifier"]. Otherwise we fall back to uniform probabilities.
        """
        # Check that the model is fitted.
        check_is_fitted(self)

        # Validate input.
        if not self.bert:
            X = check_array(X, accept_sparse="csr", allow_nd=True, ensure_2d=True)
        else:
            X = np.array(X)
        n_samples = X.shape[0]
        L = self.max_levels_  # total number of levels in the hierarchy

        # Initialize prediction array and winning probability array.
        y = np.full((n_samples, L), "", dtype=object)
        win_prob = np.full((n_samples, L), np.nan)

        # --- LEVEL 0: Decision from children of the root ---
        children0 = list(self.hierarchy_.successors(self.root_))
        if not children0:
            return y, win_prob
        n_children = len(children0)
        probs = np.zeros((n_samples, n_children))
        for i, child in enumerate(children0):
            # Try to retrieve the classifier from our dictionary...
            classifier = self.classifiers_.get(child)
            # ...and if not found, try from the node attribute.
            if classifier is None and child in self.hierarchy_.nodes():
                classifier = self.hierarchy_.nodes[child].get("classifier")
            if classifier is None:
                # Fallback: uniform probability.
                probs[:, i] = 1.0 / n_children
            else:
                # For binary classifiers assume the positive class is at index 1.
                if len(classifier.classes_) == 2:
                    pos_idx = 1
                else:
                    try:
                        pos_idx = list(classifier.classes_).index(child)
                    except ValueError:
                        pos_idx = 0
                probs[:, i] = classifier.predict_proba(X)[:, pos_idx]
        pred_indices = np.argmax(probs, axis=1)
        for i in range(n_samples):
            chosen_child = children0[pred_indices[i]]
            y[i, 0] = chosen_child
            win_prob[i, 0] = probs[i, pred_indices[i]]

        # --- LEVELS 1 to L-1: Decisions at deeper levels ---
        for level in range(1, L):
            unique_parents = np.unique(y[:, level - 1])
            for parent in unique_parents:
                indices = np.where(y[:, level - 1] == parent)[0]
                if indices.size == 0:
                    continue
                children = list(self.hierarchy_.successors(parent))
                if not children:
                    continue  # parent is a leaf node
                X_subset = X[indices]
                n_children = len(children)
                sub_probs = np.zeros((X_subset.shape[0], n_children))
                for j, child in enumerate(children):
                    classifier = self.classifiers_.get(child)
                    if classifier is None and child in self.hierarchy_.nodes():
                        classifier = self.hierarchy_.nodes[child].get("classifier")
                    if classifier is None:
                        sub_probs[:, j] = 1.0 / n_children
                    else:
                        if len(classifier.classes_) == 2:
                            pos_idx = 1
                        else:
                            try:
                                pos_idx = list(classifier.classes_).index(child)
                            except ValueError:
                                pos_idx = 0
                        sub_probs[:, j] = classifier.predict_proba(X_subset)[:, pos_idx]
                sub_pred_indices = np.argmax(sub_probs, axis=1)
                for k, sample_idx in enumerate(indices):
                    chosen_child = children[sub_pred_indices[k]]
                    y[sample_idx, level] = chosen_child
                    win_prob[sample_idx, level] = sub_probs[k, sub_pred_indices[k]]
        return win_prob


class LocalClassifierPerParentNodeWrapper(LocalClassifierPerParentNode):
    def __init__(
        self,
        local_classifier: None,
        verbose: int = 0,
        edge_list: str = "",
        replace_classifiers: bool = True,
        n_jobs: int = 1,
        bert: bool = False,
    ):
        super().__init__(
            local_classifier=local_classifier,
            verbose=verbose,
            edge_list=edge_list,
            replace_classifiers=replace_classifiers,
            n_jobs=n_jobs,
            bert=bert,
        )

    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        if not self.bert:
            X = check_array(X, accept_sparse="csr", allow_nd=True, ensure_2d=False)
        else:
            X = np.array(X)

        # Initialize array that holds predictions
        y = np.empty((X.shape[0], self.max_levels_), dtype=self.dtype_)
        # We initialize a dictionary that will hold the probabilities for each node
        # probs = np.empty((X.shape[0], self.max_levels_, ), dtype=self.dtype_)
        probs = {}

        self.logger_.info("Predicting")

        # Predict first level
        classifier = self.hierarchy_.nodes[self.root_]["classifier"]
        y[:, 0] = classifier.predict(X).flatten()
        probs["l0"] = classifier.predict_proba(X)

        self._predict_remaining_levels(X, y, probs)

        y = self._convert_to_1d(y)

        self._remove_separator(y)

        return y, probs

    def _predict_remaining_levels(self, X, y, probs=None):
        for level in range(1, y.shape[1]):
            predecessors = set(y[:, level - 1])
            predecessors.discard("")
            probs["l{}".format(level)] = {}
            for predecessor in predecessors:
                mask = np.isin(y[:, level - 1], predecessor)
                predecessor_x = X[mask]
                if predecessor_x.shape[0] > 0:
                    successors = list(self.hierarchy_.successors(predecessor))
                    if len(successors) > 0:
                        classifier = self.hierarchy_.nodes[predecessor]["classifier"]
                        y[mask, level] = classifier.predict(predecessor_x).flatten()
                        if probs is not None:
                            _probs = classifier.predict_proba(predecessor_x)
                            level_probs = np.empty((y.shape[0], _probs.shape[-1]))
                            level_probs[mask, :] = _probs
                            probs["l{}".format(level)][predecessor] = level_probs
