from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..dataset import ExperimentData
from ..dataset.ml_data import MLExperimentData
from ..utils.enums import MLMode
from .executor import Executor


class MLExecutor(Executor, ABC):
    """Base executor for MLExperiment stages.

    MLExecutor is intended to be executed only from MLExperiment. It accepts a mode
    propagated by MLExperiment and dispatches execution to mode-specific handlers.
    """

    def __init__(self, key: Any = ""):
        super().__init__(key=key)

    def execute(
        self,
        data: ExperimentData,
        mode: MLMode | None = None,
    ) -> ExperimentData:
        mode = MLMode.FIT_PREDICT if mode is None else mode
        if not isinstance(mode, MLMode):
            raise ValueError(f"Unknown mode: {mode}")

        if not isinstance(data, MLExperimentData):
            raise TypeError(
                "MLExecutor must be executed with MLExperimentData and used from MLExperiment"
            )

        if mode == MLMode.FIT:
            return self.execute_fit(data)
        if mode == MLMode.PREDICT:
            return self.execute_predict(data)
        if mode == MLMode.FIT_PREDICT:
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
