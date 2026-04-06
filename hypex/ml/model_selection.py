from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

from ..dataset.ml_data import MLData, MLExperimentData
from ..executor.ml_executor import MLExecutor
from ..utils.adapter import Adapter
from .models import MLModel


class ModelSelectionExecutor(MLExecutor):
    """Select and fit the best model per target using cross-validation."""

    def __init__(
        self,
        models: Union[str, Sequence[str], None] = None,
        key: Any = "",
        n_folds: int = 5,
        random_state: Optional[int] = None,
        cv_aggregation: str = "mean",
    ):
        super().__init__(key=key)
        self.models = models
        self.n_folds = n_folds
        self.random_state = random_state
        self.cv_aggregation = cv_aggregation

    def _validate_models(self) -> List[str]:
        available_models = list(MLModel._MODEL_BACKENDS.keys())

        if self.models is None:
            return available_models

        model_names = Adapter.to_list(self.models)
        wrong_models = [m for m in model_names if m.lower() not in available_models]
        if wrong_models:
            raise ValueError(
                f"Unsupported models: {wrong_models}. Available models: {available_models}"
            )
        return model_names

    def execute_fit(self, data: MLExperimentData) -> MLExperimentData:
        models = self._validate_models()
        for target_name in data.get_all_targets():
            ml_data_obj = data.get_ml_data(target_name)
            self._fit_best_model(data, target_name, ml_data_obj, models)
        return data

    def execute_predict(self, data: MLExperimentData) -> MLExperimentData:
        for target_name in data.get_all_targets():
            if self.id not in data.trained_models or target_name not in data.trained_models[self.id]:
                raise ValueError(
                    f"No selected model found for target '{target_name}' (executor={self.id}). "
                    "Make sure to run FIT mode first or load experiment."
                )
        return data

    def _fit_best_model(
        self,
        data: MLExperimentData,
        target: str,
        ml_data_obj: MLData,
        models: List[str],
    ) -> None:
        if self.id in data.trained_models and target in data.trained_models[self.id]:
            return

        feature_names = list(ml_data_obj.X_train.columns)
        best_model = None
        best_stats = None
        best_score = -float("inf")

        for model_name in models:
            model = MLModel.create(model_name)
            stats = model.cross_validate(
                ml_data_obj.X_train,
                ml_data_obj.Y_train,
                n_folds=self.n_folds,
                random_state=self.random_state,
                aggregation=self.cv_aggregation,
            )
            if stats.variance_reduction_cv > best_score:
                best_score = stats.variance_reduction_cv
                best_model = model
                best_stats = stats

        if best_model is None or best_stats is None:
            raise RuntimeError(
                f"No models were successfully fitted for target '{target}'. "
                "All models failed during training."
            )

        best_model.fit(ml_data_obj.X_train, ml_data_obj.Y_train)

        mapped_importances = {
            feature_names[int(col_idx)]: importance
            for col_idx, importance in best_stats.feature_importances.items()
        }
        best_stats.feature_importances = mapped_importances

        data.add_trained_model(self.id, target, best_model, best_stats)
