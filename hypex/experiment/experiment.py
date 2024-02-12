from abc import ABC, abstractmethod
from typing import Iterable
from copy import deepcopy

from hypex.dataset.dataset import Dataset, ExperimentData
from hypex.analyzer.analyzer import Analyzer

class Executor(ABC):
    full_name: str
    _id: int

    @abstractmethod
    def generate_full_name(self) -> str:
        pass

    def __init__(self, full_name: str = None):
        self.full_name = full_name or self.generate_full_name()
        self._id = id(self)

    @abstractmethod
    def execute(self, data: ExperimentData) -> ExperimentData:
        pass

class Experiment(Executor):
    def generate_full_name(self) -> str:
        return f"Experiment({len(self.executors)})"

    def __init__(self, executors: Iterable[Executor], full_name: str = None):
        self.executors : Iterable[Executor] = executors
        super().__init__(full_name)

    def execute(self, data: ExperimentData) -> ExperimentData:
        experiment_data: ExperimentData = data
        for executor in self.executors:
            experiment_data = executor.execute(experiment_data)
        return experiment_data

class CycledExperiment(Executor):
    def generate_full_name(self) -> str:
        return f"CycledExperiment({self.inner_experiment.full_name} x {self.n_iterations})"

    def __init__(self, inner_experiment: Experiment, n_iterations: int, analyzer: Analyzer, full_name: str = None):
        self.inner_experiment: Experiment = inner_experiment
        self.n_iterations: int = n_iterations
        self.analyzer: Analyzer = analyzer
        super().__init__(full_name)

    def execute(self, data: ExperimentData) -> ExperimentData:
        for _ in range(self.n_iterations):
            data = self.analyzer.execute(self.inner_experiment.execute(data))
        return data