from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import joblib

from ..dataset import ExperimentData
from ..dataset.ml_experiment_data import MLExperimentData
from ..utils.enums import MLModeEnum
from .executor import Executor


class MLExecutor(Executor, ABC):
    """Base executor for MLExperiment stages.

    MLExecutor is intended to be executed only from MLExperiment.
    """

    def __init__(self, key: Any = ""):
        super().__init__(key=key)
        self._model: Any = None
        self._model_stats: dict[str, Any] = {}

    # ── Model access ────────────────────────────────────────────

    @property
    def model(self) -> Any:
        return self._model

    @model.setter
    def model(self, value: Any) -> None:
        self._model = value

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

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
            return self.execute_fit(data)
        if mode == MLModeEnum.PREDICT:
            return self.execute_predict(data)
        if mode == MLModeEnum.FIT_PREDICT:
            return self.execute_fit_predict(data)
        raise ValueError(f"Unknown mode: {mode}")

    @abstractmethod
    def execute_fit(self, data: MLExperimentData) -> MLExperimentData:
        raise NotImplementedError

    @abstractmethod
    def execute_predict(self, data: MLExperimentData) -> MLExperimentData:
        raise NotImplementedError

    def execute_fit_predict(self, data: MLExperimentData) -> MLExperimentData:
        data = self.execute_fit(data)
        data = self.execute_predict(data)
        return data

    # ── STATUS serialization protocol (the lightweight part) ─────────────

    def get_artifact(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "module": type(self).__module__,
            "id": self.id,
            "key": self.key,
            "params": self.get_params(),
            "fitted": self.is_fitted,
            "has_model": self.has_model,
            "state": self._get_fitted_state(),
        }

    def set_artifact(self, artifact: dict[str, Any]) -> None:
        self._set_fitted_state(artifact.get("state", {}))

    def _get_fitted_state(self) -> dict[str, Any]:
        return {}

    def _set_fitted_state(self, state: dict[str, Any]) -> None:
        """Hook for subclasses: restore your trained state."""

    # ── MODEL serialization protocol (the heavy part) ───────────────

    def save_model(self, path: str) -> None:
        """Сохраняет модель на диск, если она есть."""
        if not self.has_model:
            return
        os.makedirs(path, exist_ok=True)
        self._save_model_backend(self._model, path)

    def load_model(self, path: str) -> None:
        self._model = self._load_model_backend(path)

    def _save_model_backend(self, model: Any, path: str) -> None:
        joblib.dump(model, os.path.join(path, "model.pkl"))

    def _load_model_backend(self, path: str) -> Any:
        return joblib.load(os.path.join(path, "model.pkl"))
