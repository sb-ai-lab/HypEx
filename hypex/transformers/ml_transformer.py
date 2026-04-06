from __future__ import annotations

from typing import Any

from ..dataset import Dataset
from ..dataset.ml_data import MLExperimentData
from ..executor.ml_executor import MLExecutor
from ..executor.state import MLExecutorParams


class MLTransformer(MLExecutor):
    """Transformer with fit/predict lifecycle for MLExperiment."""

    def __init__(self, key: Any = "", **calc_kwargs):
        super().__init__(key=key)
        self.calc_kwargs = calc_kwargs
        self._fitted_state: MLExecutorParams | None = None

    @property
    def _is_transformer(self):
        return True

    @property
    def is_fitted(self) -> bool:
        return self._fitted_state is not None

    @property
    def fitted_params(self) -> dict[str, Any]:
        if not self.is_fitted:
            raise ValueError(f"Transformer {self.__class__.__name__} is not fitted yet")
        return self._fitted_state.fitted_params

    @staticmethod
    def _fit(data: Dataset, **kwargs) -> dict[str, Any]:
        return {}

    @staticmethod
    def _transform(data: Dataset, fitted_params: dict[str, Any], **kwargs) -> Dataset:
        return data

    def execute_fit(self, data: MLExperimentData) -> MLExperimentData:
        fitted_params = self._fit(data.ds, **self.calc_kwargs)
        state = MLExecutorParams(
            executor_id=self.id,
            executor_class=self.__class__.__name__,
            fitted_params=fitted_params,
            metadata={"init_kwargs": self.calc_kwargs},
        )
        self._fitted_state = state
        data.add_fitted_ml_executor(self.id, state)
        return data

    def execute_predict(self, data: MLExperimentData) -> MLExperimentData:
        state = data.get_fitted_ml_executor(self.id)
        if state is None:
            raise ValueError(
                f"Transformer {self.__class__.__name__} (id={self.id}) "
                "is in 'predict' mode but no fitted state found."
            )
        self._fitted_state = state
        transformed_ds = self._transform(data.ds, state.fitted_params, **self.calc_kwargs)
        return data.copy(data=transformed_ds)

    def execute_fit_predict(self, data: MLExperimentData) -> MLExperimentData:
        data = self.execute_fit(data)
        return self.execute_predict(data)
