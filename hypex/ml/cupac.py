from __future__ import annotations

from typing import Any, Sequence

from ..dataset.dataset import Dataset, ExperimentData
from ..dataset.roles import (
    AdditionalTargetRole,
    FeatureRole,
    PreTargetRole,
    TargetRole,
)
from ..executor import MLExecutor
from ..extensions.cupac import CupacExtension
from ..utils.adapter import Adapter
from ..utils.models import CUPAC_MODELS


class CUPACExecutor(MLExecutor):
    """
    Executor that applies CUPAC (Control Using Predictions As Covariates) variance reduction technique.

    CUPAC uses machine learning models to predict target values based on historical data,
    then adjusts current targets by removing the predicted variation to reduce variance.

    Args:
        cupac_models (Union[str, Sequence[str], None]): Model(s) to use for prediction.
            If None, all available models will be tried and the best one selected.
        key (Any): Unique identifier for the executor.
        n_folds (int): Number of folds for cross-validation during model selection.
        random_state (Optional[int]): Random seed for reproducibility.
    """

    def __init__(
        self,
        cupac_models: str | Sequence[str] | None = None,
        key: Any = "",
        n_folds: int = 5,
        random_state: int | None = None,
    ):
        super().__init__(target_role=TargetRole(), key=key)
        self.cupac_models = cupac_models
        self.extension = CupacExtension(n_folds, random_state)

    def _validate_models(self) -> None:
        """
        Validate that all specified CUPAC models are supported and available for the current backend.

        Raises:
            ValueError: If any model is not recognized or not available for the current backend.
        """
        wrong_models = []
        if self.cupac_models is None:
            self.cupac_models = list(CUPAC_MODELS.keys())
            return

        self.cupac_models = Adapter.to_list(self.cupac_models)

        for model in self.cupac_models:
            if model.lower() not in CUPAC_MODELS:
                wrong_models.append(model)
            elif CUPAC_MODELS[model] is None:
                raise ValueError(
                    f"Model '{model}' is not available for the current backend"
                )

        if wrong_models:
            raise ValueError(
                f"Wrong cupac models: {wrong_models}. Available models: {list(CUPAC_MODELS.keys())}"
            )

    @staticmethod
    def _prepare_data(data: ExperimentData) -> dict[str, dict[str, list]]:
        """
        Prepare data for CUPAC by organizing temporal fields into training and prediction structures.

        This method performs complex data organization:
        1. Groups target and feature fields by their temporal lags
        2. Identifies cofounders (features used for prediction)
        3. Structures data into X_train, Y_train for model training
        4. Creates X_predict for current period adjustment (if applicable)

        Args:
            data (ExperimentData): Input experiment data with temporal roles.

        Returns:
            dict: Nested dictionary with structure:
                {target_name: {
                    'X_train': [[feature_cols_at_lag_n], ..., [feature_cols_at_lag_2]],
                    'Y_train': [target_at_lag_n-1, ..., target_at_lag_1],
                    'X_predict': [[feature_cols_at_lag_1]] (optional, only for real targets)
                }}
        """

        def agg_temporal_fields(role, data) -> dict[str, dict]:
            """
            Aggregate fields by their temporal lags.

            Returns:
                dict: {field_name: {lag: field_name_with_lag}} or {field_name: {}}
                      Empty dict means lag=0 or None (current period).
            """
            fields = {}
            searched_fields = data.field_search(
                (
                    [TargetRole(), PreTargetRole()]
                    if isinstance(role, TargetRole)
                    else role
                ),
                search_types=[int, float],
            )

            searched_lags = [
                (
                    field,
                    (
                        data.ds.roles[field].lag
                        if not isinstance(data.ds.roles[field], TargetRole)
                        else 0
                    ),
                )
                for field in searched_fields
            ]
            sorted_fields_by_lag = sorted(searched_lags, key=lambda x: x[1])
            for field, lag in sorted_fields_by_lag:
                if lag in [None, 0]:
                    fields[field] = {}
                else:
                    if data.ds.roles[field].parent not in fields:
                        fields[data.ds.roles[field].parent] = {}
                    fields[data.ds.roles[field].parent][lag] = field

            return fields

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
        targets = agg_temporal_fields(TargetRole(), data)
        features = agg_temporal_fields(FeatureRole(), data)

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

    @classmethod
    def _execute_inner_function(cls) -> None:
        pass

    @classmethod
    def _inner_function(cls) -> None:
        pass

    def calc(
        self, mode: str, model: str | Any, X: Dataset, Y: Dataset | None = None
    ) -> Any:
        if mode == "kfold_fit":
            return self.kfold_fit(model, X, Y)
        elif mode == "fit":
            return self.fit(model, X, Y)
        elif mode == "predict":
            return self.predict(model, X)

    def kfold_fit(
        self, model: str, X: Dataset, Y: Dataset
    ) -> tuple[float, dict[str, float]]:
        """Run k-fold cross-validation and return variance reduction and feature importances."""
        var_red, feature_importances = self.extension.calc(
            data=X,
            mode="kfold_fit",
            model=model,
            Y=Y,
        )

        return var_red, feature_importances

    def fit(self, model: str, X: Dataset, Y: Dataset) -> Any:
        return self.extension.calc(
            data=X,
            mode="fit",
            model=model,
            Y=Y,
        )

    def predict(self, model: Any, X: Dataset) -> Dataset:
        return self.extension.calc(
            data=X,
            mode="predict",
            model=model,
        )

    @staticmethod
    def _agg_data_from_cupac_data(
        data: ExperimentData, cupac_data_slice: list
    ) -> Dataset:
        """
        Aggregate columns from cupac_data structure into a single Dataset.

        This method handles two types of column structures:
        1. Single column: [column_name] - directly extracted
        2. Multiple lag columns: [col_lag1, col_lag2, ...] - vertically stacked

        Args:
            data: Original ExperimentData with all columns.
            cupac_data_slice: List of column specifications, where each element is:
                - [single_col_name] for non-temporal columns
                - [col_name_lag1, col_name_lag2, ...] for temporal sequences

        Returns:
            Dataset with standardized column names (0, 1, 2, ...).
        """
        res_dataset = None
        column_counter = 0

        for column in cupac_data_slice:
            if len(column) == 1:
                # Single column case: extract directly
                col_data = data.ds[column[0]]
            else:
                # Multiple lag columns: stack them vertically
                res_lag_column = None
                for lag_column in column:
                    tmp_dataset = data.ds[lag_column]
                    tmp_dataset = tmp_dataset.rename({lag_column: column[0]})
                    if res_lag_column is None:
                        res_lag_column = tmp_dataset
                    else:
                        res_lag_column = res_lag_column.append(
                            tmp_dataset, reset_index=True, axis=0
                        )
                col_data = res_lag_column

            # Standardize column names to numeric format for model training
            standard_col_name = f"{column_counter}"
            col_data = col_data.rename(
                {next(iter(col_data.columns)): standard_col_name}
            )
            column_counter += 1

            if res_dataset is None:
                res_dataset = col_data
            else:
                res_dataset = res_dataset.add_column(data=col_data)
        return res_dataset

    def execute(self, data: ExperimentData) -> ExperimentData:
        """
        Execute CUPAC variance reduction on the experiment data.

        Process:
        1. Validate models and prepare temporal data structures
        2. For each target:
            a. Try all specified models with cross-validation
            b. Select the model with best variance reduction
            c. Fit the best model on all training data
            d. Predict and adjust current target values (if applicable)
            e. Calculate variance reduction metrics
        3. Store adjusted targets and metrics in ExperimentData

        Args:
            data (ExperimentData): Input data with temporal features and targets.

        Returns:
            ExperimentData: Data with CUPAC-adjusted targets and variance reduction reports.
        """
        self._validate_models()
        cupac_data = self._prepare_data(data)
        for target, target_data in cupac_data.items():
            # Extract feature names once before data aggregation
            X_train_feature_names = [column[0] for column in target_data["X_train"]]

            X_train = self._agg_data_from_cupac_data(data, target_data["X_train"])
            Y_train = self._agg_data_from_cupac_data(data, [target_data["Y_train"]])
            best_model, best_var_red, best_feature_importances = None, None, None

            # Model selection via cross-validation
            # Feature importances are extracted during CV for efficiency
            for model in self.cupac_models:
                var_red, fold_importances = self.calc(
                    mode="kfold_fit", model=model, X=X_train, Y=Y_train
                )
                if best_var_red is None or var_red > best_var_red:
                    best_model, best_var_red = model, var_red
                    # Map standardized column names to original feature names
                    best_feature_importances = {
                        X_train_feature_names[int(col_idx)]: importance
                        for col_idx, importance in fold_importances.items()
                    }

            if best_model is None:
                raise RuntimeError(
                    f"No models were successfully fitted for target '{target}'. All models failed during training."
                )

            cupac_variance_reduction_real = None

            # Apply CUPAC adjustment to current period (if target is real, not virtual)
            # We need to fit the model on all data for prediction, but importances are already from CV
            if "X_predict" in target_data:
                fitted_model = self.calc(
                    mode="fit", model=best_model, X=X_train, Y=Y_train
                )

                X_predict = self._agg_data_from_cupac_data(
                    data, target_data["X_predict"]
                )

                prediction = self.calc(mode="predict", model=fitted_model, X=X_predict)

                # Adjust target by removing explained variation
                explained_variation = prediction - prediction.mean()
                target_cupac = data.ds[target] - explained_variation

                target_cupac = target_cupac.rename({target: f"{target}_cupac"})
                data.additional_fields = data.additional_fields.add_column(
                    data=target_cupac, role={f"{target}_cupac": AdditionalTargetRole()}
                )
                cupac_variance_reduction_real = (
                    self.extension._calculate_variance_reduction(
                        data.ds[target], target_cupac
                    )
                )

            report = {
                "cupac_best_model": best_model,
                "cupac_variance_reduction_cv": best_var_red,
                "cupac_variance_reduction_real": cupac_variance_reduction_real,
                "cupac_feature_importances": best_feature_importances,
            }
            data.analysis_tables[f"{target}_cupac_report"] = report

        return data
