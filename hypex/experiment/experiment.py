from abc import ABC, abstractmethod
from typing import Iterable, Dict, Union
from copy import deepcopy
import warnings

from hypex.dataset.dataset import Dataset, ExperimentData
from hypex.analyzer.analyzer import Analyzer
from hypex.dataset.roles import TempGroupingRole, TempTargetRole


class Executor(ABC):
    @property
    def _split_symbol(self) -> str:
        return "\u2570"

    def generate_full_name(self) -> str:
        return self.__class__.__name__

    def generate_params_hash(self) -> str:
        return ""

    def generate_id(self) -> str:
        return self._split_symbol.join(
            [
                self.__class__.__name__,
                self.params_hash.replace(self._split_symbol, "|"),
                str(self.index),
            ]
        )

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


class ComplexExecutor(ABC, Executor):
    default_inner_executors: Dict[str, Executor] = {}

    def get_inner_executors(
        self, inner_executors: Dict[str, Executor] = None
    ) -> Dict[str, Executor]:
        result = {}
        for key, executor in self.default_inner_executors.items():
            if key not in inner_executors:
                warnings.warn(
                    f"{key} executor not found in inner_executors. Will {key} will be used by default."
                )
                result[key] = executor
            else:
                result[key] = inner_executors[key]
        return inner_executors

    def __init__(
        self,
        inner_executors: Dict[str, Executor] = None,
        full_name: str = None,
        index: int = 0,
    ):
        super().__init__(full_name=full_name, index=index)
        self.inner_executors = self.get_inner_executors(inner_executors)


class Experiment(Executor):
    def generate_full_name(self) -> str:
        return f"Experiment({len(self.executors)})"

    def _detect_transformer(self) -> bool:
        return False

    def get_executor_ids(self, searched_classes=None) -> Union[Dict[type, str], List[str]]:
        if searched_classes is None:
            return [executor._id for executor in self.executors]

        searched_classes = (
            searched_classes if isinstance(searched_classes, Iterable) else [searched_classes]
        )
        for sc in searched_classes:
            return {sc: [executor._id for executor in self.executors if isinstance(executor, sc)]}


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

    def _extract_result(
        self, original_data: ExperimentData, experiment_data: ExperimentData
    ):
        return experiment_data

    def execute(self, data: ExperimentData) -> ExperimentData:
        experiment_data = deepcopy(data) if self.transformer else data
        for executor in self.executors:
            experiment_data = executor.execute(experiment_data)
        return experiment_data


class CycledExperiment(Executor):
    def __init__(
        self,
        inner_executor: Executor,
        n_iterations: int,
        analyzer: Analyzer,
        full_name: Union[str, None] = None,
        index: int = 0,
    ):
        self.inner_executor: Executor = inner_executor
        self.n_iterations: int = n_iterations
        self.analyzer: Analyzer = analyzer
        super().__init__(full_name, index)

    def generate_params_hash(self) -> str:
        return f"{self.inner_executor.full_name} x {self.n_iterations}"

    def execute(self, data: ExperimentData) -> ExperimentData:
        for _ in range(self.n_iterations):
            data = self.analyzer.execute(self.inner_executor.execute(data))
        return data


class GroupExperiment(Executor):
    def generate_params_hash(self) -> str:
        return f"{self.grop_field}->{self.inner_executor._id.replace('|', '')}"

    def __init__(
        self,
        inner_executor: Executor,
        full_name: str = None,
        index: int = 0,
    ):
        self.inner_executor: Executor = inner_executor
        super().__init__(full_name, index)

    def extract_result(self, data: ExperimentData) -> Dataset:
        return data.analysis_tables[self.inner_executor._id]

    def insert_result(
        self, data: ExperimentData, result_list: List[Dataset]
    ) -> ExperimentData:
        result = result_list[0]
        for i in range(1, len(result_list)):
            result = result.append(result_list[i])
        data.analysis_tables[self._id] = result
        return data

    def execute(self, data: ExperimentData) -> ExperimentData:
        result_list = []
        group_field = data.data.get_columns_by_roles(TempGroupingRole, tmp_role=True)

        for group, group_data in data.data.groupby(group_field):
            temp_data = ExperimentData(group_data)
            temp_data = self.inner_executor.execute(temp_data)
            result_list.append(self.extract_result(temp_data))
        return self.insert_result(data, result_list)


class OnTargetExperiment(Experiment):
    def execute(self, data: ExperimentData) -> ExperimentData:
        for field in data.data.get_columns_by_roles(TargetRole):
            data.data.tmp_roles = {field: TempTargetRole()}
            data = super().execute(data)
            data.data.tmp_roles = {}
        return data
