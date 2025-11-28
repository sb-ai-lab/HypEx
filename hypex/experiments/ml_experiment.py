from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, Sequence

from ..dataset import Dataset, ExperimentData, MLExperimentData
from ..executor import Executor
from .base import Experiment

class MLExperiment(Experiment):
    def __init__(
        self,
        executors: Sequence[Executor],
        transformer: bool | None = None,
        key: Any = "",
    ):
        super().__init__(executors, transformer, key)

    def execute(self, data: ExperimentData) -> ExperimentData:
        ml_experiment_data = MLExperimentData.from_experiment_data(data)
        
        ml_experiment_data = super.execute(ml_experiment_data)

        return ml_experiment_data.to_experiment_data()

