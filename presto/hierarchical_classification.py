from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted, check_array
from catboost import CatBoostClassifier
from hiclass import LocalClassifierPerParentNode, LocalClassifierPerNode


class CatBoostClassifierWrapper(CatBoostClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y, *args, **kwargs):
        val_fraction = 0.3
        early_stopping_rounds = 100

        _X_trn, _X_val, _y_trn, _y_val = train_test_split(X, y, stratify=y, test_size=val_fraction)

        return super().fit(
            _X_trn, _y_trn, eval_set=(_X_val, _y_val), early_stopping_rounds=early_stopping_rounds
        )


class LocalClassifierPerNodeWrapper(LocalClassifierPerNode):
    def __init__(
        self,
        local_classifier: None,
        binary_policy: str = "siblings",
        verbose: int = 0,
        edge_list: str = None,
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
            # classifier_abbreviation="LCPN",
            bert=bert,
        )
        self.binary_policy = binary_policy

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

        # TODO: Add threshold to stop prediction halfway if need be

        bfs = nx.bfs_successors(self.hierarchy_, source=self.root_)

        self.logger_.info("Predicting")

        # We initialize a dictionary that will hold the probabilities for each node
        probability_dict = {}
        for predecessor, successors in bfs:
            if predecessor == self.root_:
                mask = [True] * X.shape[0]
                subset_x = X[mask]
            else:
                mask = np.isin(y, predecessor).any(axis=1)
                subset_x = X[mask]
            if subset_x.shape[0] > 0:
                probabilities = np.zeros((subset_x.shape[0], len(successors)))
                for i, successor in enumerate(successors):
                    successor_name = str(successor).split(self.separator_)[-1]
                    self.logger_.info(f"Predicting for node '{successor_name}'")
                    classifier = self.hierarchy_.nodes[successor]["classifier"]
                    positive_index = np.where(classifier.classes_ == 1)[0]
                    probabilities[:, i] = classifier.predict_proba(subset_x)[:, positive_index][
                        :, 0
                    ]

                # For each node, save the probabilities in the dictiopnary
                probability_dict[predecessor] = probabilities
                highest_probability = np.argmax(probabilities, axis=1)
                prediction = []
                for i in highest_probability:
                    prediction.append(successors[i])
                level = nx.shortest_path_length(self.hierarchy_, self.root_, predecessor)
                prediction = np.array(prediction)
                y[mask, level] = prediction

        y = self._convert_to_1d(y)

        self._remove_separator(y)

        return y, probability_dict


class LocalClassifierPerParentNodeWrapper(LocalClassifierPerParentNode):
    def __init__(
        self,
        local_classifier: None,
        verbose: int = 0,
        edge_list: str = None,
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
