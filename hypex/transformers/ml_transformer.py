from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ..dataset import Dataset
from ..dataset.ml_data import MLExperimentData
from ..executor.ml_executor import MLExecutor
from ..utils import AbstractMethodError
from .abstract import Transformer
from .state import TransformerParams


class MLTransformer(MLExecutor, Transformer):
    """Transformer with fit/predict lifecycle for MLExperiment."""

    def __init__(self, key: Any = "", **calc_kwargs):
        super().__init__(key=key)
        self.calc_kwargs = calc_kwargs
        self._fitted_state: TransformerParams | None = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted_state is not None

    @property
    def fitted_params(self) -> dict[str, Any]:
        if not self.is_fitted:
            raise ValueError(f"Transformer {self.__class__.__name__} is not fitted yet")
        return self._fitted_state.fitted_params

    @staticmethod
    @abstractmethod
    def _inner_function(data: Dataset, **kwargs) -> Dataset:
        raise AbstractMethodError

    @staticmethod
    def _fit(data: Dataset, **kwargs) -> dict[str, Any]:
        return {}

    @staticmethod
    def _transform(data: Dataset, fitted_params: dict[str, Any], **kwargs) -> Dataset:
        return MLTransformer._inner_function(data, **kwargs)

    @classmethod
    def fit(cls, data: Dataset, **kwargs) -> TransformerParams:
        fitted_params = cls._fit(data, **kwargs)
        return TransformerParams(
            transformer_id="temp",
            transformer_class=cls.__name__,
            fitted_params=fitted_params,
            metadata={"init_kwargs": kwargs},
        )

    @classmethod
    def transform(cls, data: Dataset, fitted_params: dict[str, Any], **kwargs) -> Dataset:
        return cls._transform(data, fitted_params, **kwargs)

    @classmethod
    def fit_transform(cls, data: Dataset, **kwargs) -> tuple[Dataset, TransformerParams]:
        state = cls.fit(data, **kwargs)
        transformed = cls.transform(data, state.fitted_params, **kwargs)
        return transformed, state

    def execute_fit(self, data: MLExperimentData) -> MLExperimentData:
        state = self.fit(data.ds, **self.calc_kwargs)
        state.transformer_id = self.id
        self._fitted_state = state
        data.add_fitted_transformer(self.id, state)
        return data

    def execute_predict(self, data: MLExperimentData) -> MLExperimentData:
        state = data.get_fitted_transformer(self.id)
        if state is None:
            raise ValueError(
                f"Transformer {self.__class__.__name__} (id={self.id}) "
                "is in 'predict' mode but no fitted state found."
            )
        self._fitted_state = state
        transformed_ds = self.transform(data.ds, state.fitted_params, **self.calc_kwargs)
        return data.copy(data=transformed_ds)

    def execute_fit_predict(self, data: MLExperimentData) -> MLExperimentData:
        transformed_ds, state = self.fit_transform(data.ds, **self.calc_kwargs)
        state.transformer_id = self.id
        self._fitted_state = state
        data.add_fitted_transformer(self.id, state)
        return data.copy(data=transformed_ds)
