from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

from ..dataset import Dataset, ExperimentData
from ..dataset.ml_data import MLExperimentData
from ..dataset.roles import (
    AdditionalTargetRole,
)
from ..executor.ml_executor import MLExecutor
from ..utils import ExperimentDataEnum
from ..utils.adapter import Adapter
from .models import MLModel
from ..dataset.ml_data import MLData


class CUPACExecutor(MLExecutor):
    """
    Executor that applies CUPAC (Control Using Predictions As Covariates) variance reduction technique.

    CUPAC uses machine learning models to predict target values based on historical data,
    then adjusts current targets by removing the predicted variation to reduce variance.
    
    Supports three execution modes (set by MLExperiment):
    - fit: Only train models, save for later
    - predict: Only apply adjustment using pre-trained models
    - fit_predict: Train and apply adjustment (default)

    Args:
        cupac_models (Union[str, Sequence[str], None]): Model(s) to use for prediction.
            If None, all available models will be tried and the best one selected.
        key (Any): Unique identifier for the executor.
        n_folds (int): Number of folds for cross-validation during model selection.
        random_state (Optional[int]): Random seed for reproducibility.
        cv_aggregation (str): Method for aggregating CV scores ('mean', 'median', 'max').
        mode (str | MLMode | None): Execution mode - set by MLExperiment, defaults to fit_predict.
    """

    def __init__(
        self,
        cupac_models: Union[str, Sequence[str], None] = None,
        key: Any = "",
        n_folds: int = 5,
        random_state: Optional[int] = None,
        cv_aggregation: str = "mean",
        mode: str | None = None,
    ):
        super().__init__(key=key, mode=mode)
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

    def execute_fit(self, data: MLExperimentData) -> MLExperimentData:
        """Train CUPAC models only (fit mode)."""
        models = self._validate_models()
        for target_name in data.get_all_targets():
            ml_data_obj = data.get_ml_data(target_name)
            self._fit_models(data, target_name, ml_data_obj, models)
        return data

    def execute_predict(self, data: MLExperimentData) -> MLExperimentData:
        """Apply CUPAC adjustment using pre-trained models (predict mode)."""
        for target_name in data.get_all_targets():
            ml_data_obj = data.get_ml_data(target_name)
            self._predict_with_model(data, target_name, ml_data_obj)
        return data

    def execute_fit_predict(self, data: MLExperimentData) -> MLExperimentData:
        """Train and apply CUPAC in one pass (fit_predict mode)."""
        models = self._validate_models()
        for target_name in data.get_all_targets():
            ml_data_obj = data.get_ml_data(target_name)
            self._fit_models(data, target_name, ml_data_obj, models)
            self._predict_with_model(data, target_name, ml_data_obj)
        return data
    
    def _fit_models(
        self,
        data: MLExperimentData,
        target: str,
        ml_data_obj: "MLData",
        models: List[str],
    ) -> None:
        """
        Train models for a target and save them.
        
        This is the FIT phase - no adjustment applied yet.
        Only trains if model doesn't already exist (for idempotency).
        """
        # Check if already trained (skip if already exists)
        if self.id in data.trained_models and target in data.trained_models[self.id]:
            # Model already trained, skip
            return
        
        # Get feature names from X_train columns
        X_train_feature_names = list(ml_data_obj.X_train.columns)
        
        # Model selection via CV
        best_model = None
        best_stats = None
        best_score = -float("inf")
        
        for model_name in models:
            # Create model
            model = MLModel.create(model_name)
            
            # Cross-validate
            stats = model.cross_validate(
                ml_data_obj.X_train,
                ml_data_obj.Y_train,
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
        
        # Fit best model on all training data
        best_model.fit(ml_data_obj.X_train, ml_data_obj.Y_train)
        
        # Map feature importances to original names
        mapped_importances = {
            X_train_feature_names[int(col_idx)]: importance
            for col_idx, importance in best_stats.feature_importances.items()
        }
        best_stats.feature_importances = mapped_importances
        
        # Store model and stats (no variance_reduction_real yet in FIT mode)
        data.add_trained_model(self.id, target, best_model, best_stats)
        
        # Store report for compatibility
        report = {
            "cupac_best_model": best_stats.model_name,
            "cupac_variance_reduction_cv": best_stats.variance_reduction_cv,
            "cupac_variance_reduction_real": None,  # Not computed in FIT-only mode
            "cupac_feature_importances": best_stats.feature_importances,
        }
        data.set_value(ExperimentDataEnum.analysis_tables, f"{target}_cupac_report", report)
    
    def _predict_with_model(
        self,
        data: MLExperimentData,
        target: str,
        ml_data_obj: "MLData",
    ) -> None:
        """
        Apply CUPAC adjustment using trained model.
        
        This is the PREDICT phase - uses pre-trained model.
        """
        # Get trained model
        if self.id not in data.trained_models or target not in data.trained_models[self.id]:
            raise ValueError(
                f"No trained model found for target '{target}' (executor={self.id}). "
                f"Make sure to run FIT mode first or load experiment. "
                f"Available models: {list(data.trained_models.keys())}"
            )
        
        model = data.trained_models[self.id][target]
        stats = data.model_stats[self.id][target]
        
        # Apply CUPAC adjustment only if we have prediction data
        if ml_data_obj.X_predict is not None:
            var_red_real = self._apply_cupac_adjustment(
                data=data,
                model=model,
                target=target,
                ml_data_obj=ml_data_obj
            )
            
            # Update stats with real variance reduction
            stats.variance_reduction_real = var_red_real
            data.add_trained_model(self.id, target, model, stats)
            
            # Update report
            report = {
                "cupac_best_model": stats.model_name,
                "cupac_variance_reduction_cv": stats.variance_reduction_cv,
                "cupac_variance_reduction_real": stats.variance_reduction_real,
                "cupac_feature_importances": stats.feature_importances,
            }
            data.set_value(ExperimentDataEnum.analysis_tables, f"{target}_cupac_report", report)
        else:
            # No current period data - cannot apply adjustment
            # This is valid in FIT mode with only historical data
            pass
    
    def _apply_cupac_adjustment(
        self,
        data: MLExperimentData,
        model: MLModel,
        target: str,
        ml_data_obj: "MLData",
    ) -> float:
        """Apply CUPAC adjustment to current period"""
        prediction_ds = model.predict(ml_data_obj.X_predict)
        
        # Get prediction mean using Dataset API
        prediction_mean = float(prediction_ds.mean())
        
        # Get target and prediction as Datasets
        target_ds = data.ds[target]
        
        # Perform adjustment using Dataset arithmetic operations
        # adjusted = target - prediction + prediction_mean
        adjusted_ds = target_ds - prediction_ds["prediction"] + prediction_mean
        
        # Rename to cupac column
        adjusted_ds = adjusted_ds.rename({target: f"{target}_cupac"})
        
        # Update role to AdditionalTargetRole
        adjusted_ds.roles[f"{target}_cupac"] = AdditionalTargetRole()
        
        # Add to additional_fields
        data.additional_fields = data.additional_fields.add_column(data=adjusted_ds)
        
        # Calculate variance reduction using Dataset values
        target_values = target_ds.get_values(column=target)
        adjusted_values = adjusted_ds.get_values(column=f"{target}_cupac")
        var_red = model._calculate_variance_reduction(target_values, adjusted_values)
        return var_red

    @staticmethod
    def _agg_data_from_cupac_data(
        data: ExperimentData, cupac_data_slice: list
    ) -> Dataset:
        """
        Aggregate columns from cupac_data structure into a single Dataset.
        
        DEPRECATED: This method is kept for backward compatibility.
        New code should use MLData directly from CUPACDataSplitter.

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
