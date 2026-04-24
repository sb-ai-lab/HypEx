from __future__ import annotations

from typing import Any, Optional

from ..dataset import Dataset, ExperimentData
from ..dataset.ml_data import MLData, MLExperimentData
from ..dataset.roles import FeatureRole, PreTargetRole, TargetRole
from ..utils.enums import MLModeEnum
from .base import MLSplitter


class CUPACDataSplitter(MLSplitter):
    """
    Splitter that prepares temporal data structures for CUPAC ML models.
    
    This splitter organizes temporal fields (targets and features with lags) into
    MLData objects that encapsulate all data needed for ML model training.
    
    Process:
    1. Groups target and feature fields by their temporal lags
    2. Identifies cofounders (features used for prediction)
    3. Creates MLData for each target with:
       - X_train, Y_train: aggregated training data from all lags
       - X_predict: current period features (optional, for real targets only)
       - crossval: optional CV folds if generate_cv_folds=True
    
    The results are stored in MLExperimentData.ml as {target_name: MLData}.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        random_state: Optional[int] = None,
        generate_cv_folds: bool = False,
        key: Any = "",
    ):
        """
        Initialize CUPAC data splitter.
        
        Args:
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            generate_cv_folds: Whether to generate CV folds for each target
            key: Executor identifier
        """
        super().__init__(n_folds=n_folds, random_state=random_state, key=key)
        self.generate_cv_folds = generate_cv_folds
    
    def execute(
        self,
        data: MLExperimentData,
        mode: MLModeEnum | None = None,
    ) -> MLExperimentData:
        """
        Execute CUPAC data splitting.
        
        Organizes temporal data and creates MLData objects for each target.
        
        Args:
            data: ExperimentData with temporal roles
            
        Returns:
            MLExperimentData with MLData objects in .ml
        """
        mode = MLModeEnum.FIT_PREDICT if mode is None else mode
        if not isinstance(mode, MLModeEnum):
            raise ValueError(f"Unknown mode: {mode}")

        # CUPAC splitter is ML-only and must run inside MLExperiment.
        ml_data = data
        
        # Prepare raw splits structure
        raw_splits = self._prepare_splits(ml_data)

        return self.build_mldata_from_splits(
            data=ml_data,
            raw_splits=raw_splits,
            generate_cv_folds=self.generate_cv_folds,
        )

    @staticmethod
    def _build_train_groups(data: ExperimentData, y_train_columns: list[str]) -> list:
        """Build row-aligned group labels for training rows from lagged target columns."""
        groups = []
        for column_name in y_train_columns:
            groups.extend(list(data.ds[column_name].index))
        return groups

    def build_mldata_from_splits(
        self,
        data: MLExperimentData,
        raw_splits: dict[str, dict[str, list]],
        generate_cv_folds: bool = False,
    ) -> MLExperimentData:
        """
        Build MLData objects for CUPAC and attach groups created at lag parsing stage.

        Group labels are derived from the original row indexes of lagged training target
        columns, so all rows for the same user across lags share one fold group.
        """
        for target_name, target_splits in raw_splits.items():
            X_train = self.aggregate_columns(data, target_splits["X_train"])
            Y_train = self.aggregate_columns(data, [target_splits["Y_train"]])

            groups = self._build_train_groups(data, target_splits["Y_train"])
            if len(groups) != X_train.shape[0] or len(groups) != Y_train.shape[0]:
                raise ValueError(
                    "Invalid CUPAC grouping: groups, X_train and Y_train must have the same number of rows."
                )

            X_predict = None
            if "X_predict" in target_splits:
                X_predict = self.aggregate_columns(data, target_splits["X_predict"])

            ml_data_obj = MLData(
                X_train=X_train,
                Y_train=Y_train,
                X_predict=X_predict,
                groups=groups,
            )

            if generate_cv_folds:
                ml_data_obj = self.add_cv_folds_to_mldata(
                    ml_data_obj,
                    n_folds=self.n_folds,
                    random_state=self.random_state,
                )

            data.add_ml_data(target_name, ml_data_obj)

        return data
    
    def _prepare_splits(self, data: ExperimentData) -> dict[str, dict[str, list]]:
        """
        Prepare raw data splits structure (backward compatible with old implementation).
        
        Returns:
            dict: {target_name: {
                'X_train': [[feature_cols_at_lag_n], ..., [feature_cols_at_lag_2]],
                'Y_train': [target_at_lag_n-1, ..., target_at_lag_1],
                'X_predict': [[feature_cols_at_lag_1]] (optional)
            }}
        """

        def agg_train_predict_x(mode: str, lag: int) -> None:
            """
            Aggregate features and targets for a specific lag into training/prediction sets.

            For each cofounder feature, creates a list structure where:
            - First and last lags start new sublists
            - Intermediate lags append to existing sublists
            This groups temporal sequences of the same feature together.
            """
            for i, cofounder in enumerate(cofounders[target]):
                if lag in [1, max_lags[target]]:
                    cupac_data[target][mode].append([features[cofounder][lag]])
                else:
                    cupac_data[target][mode][i].append(cofounder)

            cupac_data[target][mode].append([targets[target][lag]])

        cupac_data = {}
        targets = self.aggregate_fields_by_lag(data, TargetRole())
        features = self.aggregate_fields_by_lag(data, FeatureRole())

        # Determine cofounders (features used for prediction) for each target
        cofounders = {}
        for target in targets:
            if target in data.ds.columns:
                cofounders[target] = data.ds.roles[target].cofounders
            else:
                # For virtual targets, get cofounders from the earliest lag
                min_lag = min(targets[target].keys())
                cofounders[target] = data.ds.roles[targets[target][min_lag]].cofounders

                if cofounders[target] is None:
                    raise ValueError(
                        f"Cofounders must be defined in the first lag for virtual target '{target}'"
                    )

        # Calculate maximum lag for each target (max across target lags and cofounder feature lags)
        max_lags = {}
        for target, lags in targets.items():
            if lags:
                max_lag = max(lags.keys())
                for feature in cofounders[target]:
                    if features.get(feature):
                        max_lag = max(max(features[feature].keys()), max_lag)
            max_lags[target] = max_lag

        # Build training and prediction structures for each target
        for target in targets.keys():

            cupac_data[target] = {"X_train": [], "Y_train": []}
            # Only real targets (not virtual) need prediction
            if target in data.ds.columns:
                cupac_data[target]["X_predict"] = []

            # Build training data: iterate from max_lag down to 2
            # Each iteration creates X_train entry for lag and Y_train entry for lag-1
            for lag in range(max_lags[target], 1, -1):
                agg_train_predict_x("X_train", lag)
                cupac_data[target]["Y_train"].append(targets[target][lag - 1])

            # Build prediction data for current period (lag=1) if applicable
            if "X_predict" in cupac_data[target].keys():
                agg_train_predict_x("X_predict", 1)

        return cupac_data
