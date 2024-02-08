from abc import ABC, abstractmethod
from typing import Iterable
from copy import deepcopy

from hypex.dataset.base import Dataset

class Executor(ABC):
    pass

    @abstractmethod
    def execute(self, data: Dataset) -> Dataset:
        pass

class Experiment(Executor):

    def __init__(self, executors: Iterable[Executor]):
        self.executors : Iterable[Executor] = executors

    def execute(self, data: Dataset) -> Dataset:
        experiment_data: Dataset = data
        for executor in self.executors:
            experiment_data = executor.execute(experiment_data)
        return experiment_data

class CycledExperiment(Executor):
    def __init__(self, inner_experiment: Experiment, n_iterations: int, analyzer: Analyzer):
        self.inner_experiment: Experiment = inner_experiment
        self.n_iterations: int = n_iterations
        self.analyzer: Analyzer = analyzer

    def execute(self, data: Dataset) -> Dataset:
        for _ in range(self.n_iterations):
            data = self.analyzer.execute(self.inner_experiment.execute(data))
        return data
    

print(Experiment(None).id)