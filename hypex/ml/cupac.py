from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ..dataset import Dataset, ExperimentData
from ..dataset.ml_data import MLExperimentData
from ..dataset.roles import (
    AdditionalTargetRole,
    FeatureRole,
    PreTargetRole,
    TargetRole,
)
from ..executor import Executor
from ..utils.adapter import Adapter
from .models import MLModel
from .stats import ModelStats


class CUPACExecutor(Executor):
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
        cv_aggregation (str): Method for aggregating CV scores ('mean', 'median', 'max').
    """

    def __init__(
        self,
        cupac_models: Union[str, Sequence[str], None] = None,
        key: Any = "",
        n_folds: int = 5,
        random_state: Optional[int] = None,
        cv_aggregation: str = "mean",
    ):
        super().__init__(key=key)
        self.cupac_models = cupac_models
        self.n_folds = n_folds
        self.random_state = random_state
        self.cv_aggregation = cv_aggregation

    def _validate_models(self) -> List[str]:
        """
        Validate that all specified CUPAC models are supported.

        Returns:
            List of validated model names

        Raises:
            ValueError: If any model is not recognized.
        """
        available_models = list(MLModel._MODEL_BACKENDS.keys())
        
        if self.cupac_models is None:
            return available_models
        
        models = Adapter.to_list(self.cupac_models)
        wrong_models = [m for m in models if m.lower() not in available_models]
        
        if wrong_models:
            raise ValueError(
                f"Wrong cupac models: {wrong_models}. Available models: {available_models}"
            )
        
        return models

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
        # Ensure MLExperimentData
        is_ml_data = isinstance(data, MLExperimentData)
        ml_data = (
            data if is_ml_data else MLExperimentData.from_experiment_data(data)
        )
        
        # Validate models
        models = self._validate_models()
        
        # Prepare CUPAC data structures
        cupac_data = self._prepare_data(ml_data)
        
        # Process each target
        for target, target_data in cupac_data.items():
            self._process_target(ml_data, target, target_data, models)
        
        # Return appropriate type
        return ml_data if is_ml_data else ml_data.to_experiment_data()
    
    def _process_target(
        self,
        data: MLExperimentData,
        target: str,
        target_data: Dict,
        models: List[str],
    ) -> None:
        """Process single target with model selection and adjustment"""
        
        # Check if we should load pre-trained model
        load_models_dir = data.ml.get("config", {}).get("load_models_dir")
        
        if load_models_dir:
            # Load pre-trained model instead of training
            self._load_and_apply_model(data, target, target_data, load_models_dir)
            return
        
        # Original training flow
        # Extract feature names
        X_train_feature_names = [column[0] for column in target_data["X_train"]]
        
        # Prepare training data
        X_train = self._agg_data_from_cupac_data(data, target_data["X_train"])
        Y_train = self._agg_data_from_cupac_data(data, [target_data["Y_train"]])
        
        # Model selection via CV
        best_model = None
        best_stats = None
        best_score = -float("inf")
        
        for model_name in models:
            # Create model
            model = MLModel.create(model_name)
            
            # Cross-validate (возвращает ModelStats)
            stats = model.cross_validate(
                X_train,
                Y_train,
                n_folds=self.n_folds,
                random_state=self.random_state,
                aggregation=self.cv_aggregation,
            )
            
            # Select best by variance reduction
            if stats.variance_reduction_cv > best_score:
                best_score = stats.variance_reduction_cv
                best_model = model
                best_stats = stats
        
        if best_model is None:
            raise RuntimeError(
                f"No models were successfully fitted for target '{target}'. "
                "All models failed during training."
            )
        
        # Fit best model on all data
        best_model.fit(X_train, Y_train)
        
        # Map feature importances to original names
        mapped_importances = {
            X_train_feature_names[int(col_idx)]: importance
            for col_idx, importance in best_stats.feature_importances.items()
        }
        best_stats.feature_importances = mapped_importances
        
        # Apply CUPAC adjustment if target is real
        if "X_predict" in target_data:
            var_red_real = self._apply_cupac_adjustment(
                data, best_model, target, target_data
            )
            # Update stats with real variance reduction
            best_stats.variance_reduction_real = var_red_real
        
        # Store model and stats using executor ID
        data.add_trained_model(self.id, target, best_model, best_stats)
        
        # Also store in old format for compatibility
        report = {
            "cupac_best_model": best_stats.model_name,
            "cupac_variance_reduction_cv": best_stats.variance_reduction_cv,
            "cupac_variance_reduction_real": best_stats.variance_reduction_real,
            "cupac_feature_importances": best_stats.feature_importances,
        }
        data.analysis_tables[f"{target}_cupac_report"] = report
    
    def _apply_cupac_adjustment(
        self,
        data: MLExperimentData,
        model: MLModel,
        target: str,
        target_data: Dict,
    ) -> float:
        """Apply CUPAC adjustment to current period"""
        X_predict = self._agg_data_from_cupac_data(data, target_data["X_predict"])
        
        # Predict
        prediction = model.predict(X_predict)
        prediction_mean = prediction.mean()
        
        # Adjust target
        target_values = data.ds[target].backend.data.values.ravel()
        adjusted_values = target_values - prediction + prediction_mean
        
        # Create adjusted dataset
        import pandas as pd
        
        target_cupac_df = pd.DataFrame(
            {f"{target}_cupac": adjusted_values}, index=data.ds.index
        )
        target_cupac = Dataset(
            data=target_cupac_df, roles={f"{target}_cupac": AdditionalTargetRole()}
        )
        
        # Add to additional_fields
        data.additional_fields = data.additional_fields.add_column(data=target_cupac)
        
        # Calculate variance reduction
        var_red = model._calculate_variance_reduction(target_values, adjusted_values)
        return var_red
    
    def _load_and_apply_model(
        self,
        data: MLExperimentData,
        target: str,
        target_data: TargetData,
        load_models_dir: str,
    ) -> None:
        """Load pre-trained model and apply CUPAC adjustment"""
        import os
        from .models import MLModel
        from .stats import ModelStats
        
        # Find CUPAC executor directory (starts with "CUPACExecutor")
        executor_dirs = [d for d in os.listdir(load_models_dir) 
                        if os.path.isdir(os.path.join(load_models_dir, d)) 
                        and d.startswith("CUPACExecutor")]
        
        if not executor_dirs:
            raise FileNotFoundError(
                f"No CUPACExecutor directories found in {load_models_dir}. "
                f"Make sure you saved models with save_cupac_models=True."
            )
        
        # Use the first (and should be only) CUPAC executor directory
        executor_dir = executor_dirs[0]
        model_path = os.path.join(load_models_dir, executor_dir, target)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pre-trained model for target '{target}' not found at: {model_path}. "
                f"Available targets: {os.listdir(os.path.join(load_models_dir, executor_dir))}"
            )
        
        # Load model
        model = MLModel.load(model_path)
        
        # Load stats (stored as JSON)
        stats_path = os.path.join(model_path, "stats.json")
        if os.path.exists(stats_path):
            import json
            with open(stats_path, 'r') as f:
                stats_dict = json.load(f)
            stats = ModelStats.from_dict(stats_dict)
        else:
            # Create minimal stats if not found
            stats = ModelStats(
                model_name=model._backend.__class__.__name__,
                model_type=model._backend.__class__.__name__.replace("Backend", "").lower(),
                feature_importances={},
                training_time_seconds=0.0,
            )
        
        # Apply CUPAC adjustment if target is real (has current period data)
        if "X_predict" in target_data:
            var_red_real = self._apply_cupac_adjustment(
                data, model, target, target_data
            )
            stats.variance_reduction_real = var_red_real
        
        # Store model and stats
        data.add_trained_model(self.id, target, model, stats)
        
        # Store report for compatibility
        report = {
            "cupac_best_model": stats.model_name,
            "cupac_variance_reduction_cv": stats.variance_reduction_cv,
            "cupac_variance_reduction_real": stats.variance_reduction_real,
            "cupac_feature_importances": stats.feature_importances,
        }
        data.analysis_tables[f"{target}_cupac_report"] = report

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

