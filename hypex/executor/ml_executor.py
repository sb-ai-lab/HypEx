from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..dataset import ExperimentData
from ..dataset.ml_data import MLExperimentData
from .executor import Executor


class MLExecutor(Executor, ABC):
    """Base executor for MLExperiment stages.

    MLExecutor is intended to be executed only from MLExperiment. It accepts a mode
    propagated by MLExperiment and dispatches execution to mode-specific handlers.
    """

    VALID_MODES = {"fit", "predict", "fit_predict"}

    def __init__(self, key: Any = "", mode: str | Any | None = None):
        super().__init__(key=key)
        self.mode = self._normalize_mode(mode)

    @classmethod
    def _normalize_mode(cls, mode: str | Any | None) -> str:
        if mode is None:
            return "fit_predict"

        # Support enum values (e.g. MLMode.FIT) propagated by MLExperiment.
        if hasattr(mode, "value"):
            mode = mode.value

        mode = str(mode)
        if mode not in cls.VALID_MODES:
            raise ValueError(f"Unknown mode: {mode}")
        return mode

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str | Any):
        self._mode = self._normalize_mode(value)

    def execute(self, data: ExperimentData) -> ExperimentData:
        if not isinstance(data, MLExperimentData):
            raise TypeError(
                "MLExecutor must be executed with MLExperimentData and used from MLExperiment"
            )

        if self.mode == "fit":
            return self.execute_fit(data)
        if self.mode == "predict":
            return self.execute_predict(data)
        if self.mode == "fit_predict":
            return self.execute_fit_predict(data)
        raise ValueError(f"Unknown mode: {self.mode}")

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
