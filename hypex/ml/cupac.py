from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from ..dataset import Dataset, ExperimentData
from ..dataset.ml_data import MLExperimentData
from ..dataset.roles import (
    AdditionalTargetRole,
)
from ..executor.ml_executor import MLExecutor
from ..utils import ExperimentDataEnum
from .models import MLModel
from ..dataset.ml_data import MLData


class CUPACExecutor(MLExecutor):
    """
    Executor that applies CUPAC (Control Using Predictions As Covariates)
    variance reduction technique.

    CUPAC uses a preselected machine learning model to predict target values based on
    historical data, then adjusts current targets by removing the predicted variation
    to reduce variance.
    
    Supports three execution modes (set by MLExperiment):
    - fit: Only train models, save for later
    - predict: Only apply adjustment using pre-trained models
    - fit_predict: Train and apply adjustment (default)

    Model selection is intentionally not performed in this executor. It should be
    performed by a dedicated selector executor executed earlier in the same
    MLExperiment chain.

    Args:
        cupac_models: Kept for backward compatibility with previous constructor API.
        key: Unique identifier for the executor.
        n_folds: Kept for backward compatibility.
        random_state: Kept for backward compatibility.
        cv_aggregation: Kept for backward compatibility.
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

    def execute_fit(self, data: MLExperimentData) -> MLExperimentData:
        """Validate selected models and write initial CUPAC report (fit mode)."""
        for target_name in data.get_all_targets():
            selected_executor_id, _, stats = data.get_best_model_for_target(target_name)
            report = {
                "cupac_best_model": stats.model_name,
                "cupac_variance_reduction_cv": stats.variance_reduction_cv,
                "cupac_variance_reduction_real": None,
                "cupac_feature_importances": stats.feature_importances,
                "cupac_model_executor": selected_executor_id,
            }
            data.set_value(
                ExperimentDataEnum.analysis_tables,
                f"{target_name}_cupac_report",
                report,
            )
        return data

    def execute_predict(self, data: MLExperimentData) -> MLExperimentData:
        """Apply CUPAC adjustment using pre-trained models (predict mode)."""
        for target_name in data.get_all_targets():
            ml_data_obj = data.get_ml_data(target_name)
            self._predict_with_model(data, target_name, ml_data_obj)
        return data

    def execute_fit_predict(self, data: MLExperimentData) -> MLExperimentData:
        """Apply CUPAC after selected models are already trained."""
        data = self.execute_fit(data)
        data = self.execute_predict(data)
        return data
    
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
        selected_executor_id, model, stats = data.get_best_model_for_target(target)
        
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
            data.add_trained_model(selected_executor_id, target, model, stats)
            
            # Update report
            report = {
                "cupac_best_model": stats.model_name,
                "cupac_variance_reduction_cv": stats.variance_reduction_cv,
                "cupac_variance_reduction_real": stats.variance_reduction_real,
                "cupac_feature_importances": stats.feature_importances,
                "cupac_model_executor": selected_executor_id,
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

        cov_xy = ((prediction_ds - prediction_mean) * (target_ds - target_ds.mean())).mean()
        var_x = ((prediction_ds - prediction_mean) ** 2).mean()

        if var_x == 0 or var_x != var_x:
            theta = 0
        else:
            theta = cov_xy / var_x
            
        adjusted_ds = target_ds - (prediction_ds["prediction"] - prediction_mean) * theta
        
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
