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

    def execute(self, data: ExperimentData) -> ExperimentData:
        """
        Execute CUPAC variance reduction on the experiment data.

        Process:
        1. Validate models and read prepared data from splitter
        2. For each target:
            a. Try all specified models with cross-validation
            b. Select the model with best variance reduction
            c. Fit the best model on all training data
            d. Predict and adjust current target values (if applicable)
            e. Calculate variance reduction metrics
        3. Store adjusted targets and metrics in ExperimentData

        Args:
            data (ExperimentData): Input data with splits from CUPACDataSplitter.

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
        
        # Read prepared data from splitter (stored in ml["splits"])
        cupac_data = ml_data.ml.get("splits")
        
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
        
        # Check if we should load pre-trained model from artifact
        trained_models = data.ml.get('trained_models', {})
        model_stats = data.ml.get('model_stats', {})
        
        if self.id in trained_models and target in trained_models[self.id]:
            # Use model from loaded artifact
            model = trained_models[self.id][target]
            stats = model_stats[self.id][target]
            
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
        
        prediction_ds = model.predict(X_predict)
        prediction = np.array(prediction_ds.get_values(column="prediction"))
        
        prediction_mean = float(prediction_ds.mean())
        
        target_ds = data.ds[target]
        target_values = np.array(target_ds.get_values(column=target))
        adjusted_values = target_values - prediction + prediction_mean
        
        target_cupac = Dataset.from_dict(
            data={f"{target}_cupac": adjusted_values},
            roles={f"{target}_cupac": AdditionalTargetRole()},
            index=data.ds.index
        )
        
        # Add to additional_fields
        data.additional_fields = data.additional_fields.add_column(data=target_cupac)
        
        # Calculate variance reduction
        var_red = model._calculate_variance_reduction(target_values, adjusted_values)
        return var_red

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

