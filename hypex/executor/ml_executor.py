from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import joblib

from ..dataset import ExperimentData, MLExperimentData

# from ..ml import MLExecutorState
from ..utils.enums import MLModeEnum
from .executor import Executor


@dataclass
class MLExecutorState:
    qualname: str
    module: str
    id: str
    key: str
    params: dict[str, Any]
    fitted: bool
    has_model: bool

    @property
    def json(self):
        return self.__dict__

class MLExecutor(Executor, ABC):
    """Base executor for MLExperiment stages.

    MLExecutor is intended to be executed only from MLExperiment.
    """

    def __init__(self, key: Any = "", **kwargs):
        super().__init__(key=key, **kwargs)
        self._model: Any = None
        self._model_stats: dict[str, Any] = {}

        self.is_fitted: bool = False

    # ── Model access ────────────────────────────────────────────

    @property
    def model(self) -> Any:
        return self._model

    @model.setter
    def model(self, value: Any) -> None:
        self._model = value

    @property
    def fitted(self) -> bool:
        return self.is_fitted

    @property
    def has_model(self) -> bool:
        return self._model is not None

    # ── execution ─────────────────────────────────

    def execute(
        self,
        data: ExperimentData,
        mode: MLModeEnum | None = None,
    ) -> ExperimentData:
        mode = MLModeEnum.FIT_PREDICT if mode is None else mode
        if not isinstance(mode, MLModeEnum):
            raise ValueError(f"Unknown mode: {mode}")

        if not isinstance(data, MLExperimentData):
            raise TypeError(
                "MLExecutor must be executed with MLExperimentData and used from MLExperiment"
            )

        if mode == MLModeEnum.FIT:
            return self.fit(data)
        if mode == MLModeEnum.PREDICT:
            return self.predict(data)
        if mode == MLModeEnum.FIT_PREDICT:
            return self.fit_predict(data)
        raise ValueError(f"Unknown mode: {mode}")

    @abstractmethod
    def fit(self, data: MLExperimentData) -> MLExperimentData:
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: MLExperimentData) -> MLExperimentData:
        raise NotImplementedError

    def fit_predict(self, data: MLExperimentData) -> MLExperimentData:
        data = self.execute_fit(data)
        data = self.execute_predict(data)
        return data

    # ── STATUS serialization protocol (the lightweight part) ─────────────

    def get_artifact(self) -> MLExecutorState:
        cls = type(self)
        return MLExecutorState(
            qualname=cls.__qualname__,
            module=cls.__module__,
            id=self.id,
            key=self.key,
            params=self.get_params(),
            fitted=self.fitted,
            has_model=self.has_model,
        )

    # ── MODEL serialization protocol (the heavy part) ───────────────

    def save_model(self, path: str, prefix: str="") -> None:
        """Сохраняет модель на диск, если она есть."""
        if not self.has_model:
            return
        self._save_model_backend(self._model, path, prefix)

    def load_model(self, path: str, prefix: str="") -> None:
        self._model = self._load_model_backend(path, prefix)

    def _save_model_backend(self, model: Any, path: str, prefix: str) -> None:
        model_name = self.id + "_" + prefix
        joblib.dump(model, os.path.join(path, f"{model_name}.pkl"))

    def _load_model_backend(self, path: str, prefix: str) -> Any:
        model_name = self.id + "_" + prefix
        return joblib.load(os.path.join(path, f"{model_name}.pkl"))

