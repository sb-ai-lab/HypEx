from __future__ import annotations

from typing import Any, Literal, Sequence

from ..dataset import Dataset, ExperimentData
from ..executor import Executor
from ..reporters import Reporter
from .base import Experiment
from .base_complex import ExperimentWithReporter


class MLExperiment(Experiment):
    def __init__(
        self,
        executors: Sequence[Executor],
        mode: Literal["train", "test", "val", "all"] = "all",
        transformer: bool | None = None,
        key: Any = "",
    ):
        super().__init__(executors, transformer, key)
        self.mode = mode

    def execute(self, data: ExperimentData) -> ExperimentData:
        experiment_data = data

        for executor in self.executors:
            executor.key = self.key
            experiment_data = executor.execute(experiment_data)

        return experiment_data


class MLExperimentWithReporter(ExperimentWithReporter):
    def __init__(
        self,
        executors: Sequence[Executor],
        reporter: Reporter,
        mode: Literal["train", "test", "val"] = "train",
        transformer: bool | None = None,
        key: str = "",
    ):
        super().__init__(executors, reporter, transformer, key)
        self.mode = mode

    def execute(self, data: ExperimentData) -> ExperimentData:
        experiment_data = data

        for executor in self.executors:
            executor.key = self.key
            experiment_data = executor.execute(experiment_data)

        result = self.reporter.report(experiment_data)
        return self._set_value(data, result)
