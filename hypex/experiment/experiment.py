from abc import ABC, abstractmethod
from typing import Iterable
from copy import deepcopy

from hypex.dataset.dataset import Dataset, ExperimentData
from hypex.analyzer.analyzer import Analyzer


class Executor(ABC):
    def generate_full_name(self) -> str:
        return self.__class__.__name__

    def generate_params_hash(self) -> str:
        return ""

    def generate_id(self) -> str:
        return "|".join([self.full_name, self.params_hash, str(self.index)])

    def __init__(self, full_name: str = None, index: int = 0):
        self.full_name = full_name or self.generate_full_name()
        self.index = index
        self.params_hash = self.generate_params_hash()
        self._id = generate_id()

    @property
    def _is_transformer(self) -> bool:
        return False

    @abstractmethod
    def _set_value(self, data: ExperimentData, value) -> ExperimentData:
        raise NotImplementedError

    @abstractmethod
    def execute(self, data: ExperimentData) -> ExperimentData:
        raise NotImplementedError


class Experiment(ABC, Executor):
    def generate_full_name(self) -> str:
        return f"Experiment({len(self.executors)})"

    def _detect_transformer(self) -> bool:
        return False

    def __init__(
        self,
        executors: Iterable[Executor],
        transformer: bool = None,
        full_name: str = None,
        index: int = 0,
    ):
        self.executors: Iterable[Executor] = executors
        self.transformer: bool = (
            transformer if transformer is not None else self.__detect_transformer()
        )
        super().__init__(full_name, index)

    def _extract_result(self, original_data: ExperimentData, experiment_data: ExperimentData):
        return experiment_data

    def execute(self, data: ExperimentData) -> ExperimentData:
        experiment_data = deepcopy(data) if self.transformer else data
        for executor in self.executors:
            experiment_data = executor.execute(experiment_data)
        return experiment_data


class CycledExperiment(Executor):
    def generate_full_name(self) -> str:
        return (
            f"CycledExperiment({self.inner_experiment.full_name} x {self.n_iterations})"
        )

    def __init__(
        self,
        inner_experiment: Experiment,
        n_iterations: int,
        analyzer: Analyzer,
        full_name: str = None,
        index: int = 0,
    ):
        self.inner_experiment: Experiment = inner_experiment
        self.n_iterations: int = n_iterations
        self.analyzer: Analyzer = analyzer
        super().__init__(full_name, index)

    def execute(self, data: ExperimentData) -> ExperimentData:
        for _ in range(self.n_iterations):
            data = self.analyzer.execute(self.inner_experiment.execute(data))
        return data
