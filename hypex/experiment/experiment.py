import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Iterable, Dict, Union, Any, List

from hypex.dataset.dataset import ExperimentData, Dataset
from hypex.dataset.roles import TempGroupingRole, TempTargetRole, ABCRole
from hypex.utils.constants import ID_SPLIT_SYMBOL


class Executor(ABC):
    def _generate_params_hash(self):
        self._params_hash = ""

    def _generate_id(self):
        self._id = ID_SPLIT_SYMBOL.join(
            [
                self.__class__.__name__,
                self.params_hash.replace(ID_SPLIT_SYMBOL, "|"),
                str(self._key).replace(ID_SPLIT_SYMBOL, "|"),
            ]
        )

    @property
    def id(self) -> str:
        return self._id

    @property
    def full_name(self) -> Union[str, None]:
        return self._full_name

    @full_name.setter
    def full_name(self, value: Union[str, None]):
        self._full_name: str = str(value or self.__class__.__name__)

    @property
    def key(self) -> Any:
        return self._key

    @key.setter
    def key(self, value: Any):
        self._key = value
        self._generate_id()

    @property
    def params_hash(self) -> str:
        return self._params_hash

    def refresh_params_hash(self):
        self._generate_params_hash()
        self._generate_id()

    def __init__(self, full_name: Union[str, None] = None, key: Any = ""):
        self._id: str = ""
        self._params_hash = ""
        self.full_name = full_name

        self.key: Any = key
        self.refresh_params_hash()

    @property
    def _is_transformer(self) -> bool:
        return False

    def _set_value(
        self, data: ExperimentData, value: Any = None, key: Any = None
    ) -> ExperimentData:
        return data

    @abstractmethod
    def execute(self, data: ExperimentData) -> ExperimentData:
        raise NotImplementedError


class ComplexExecutor(Executor, ABC):
    default_inner_executors: Dict[str, Executor] = {}

    def _get_inner_executors(
        self, inner_executors: Union[Dict[str, Executor], None] = None
    ) -> Dict[str, Executor]:
        result = {}
        inner_executors = inner_executors or {}
        for key, executor in self.default_inner_executors.items():
            if key not in inner_executors:
                if len(inner_executors):
                    warnings.warn(
                        f"{key} executor not found in inner_executors. Will {key} will be used by default."
                    )
                result[key] = executor
            else:
                result[key] = inner_executors[key]
        return result

    def __init__(
        self,
        inner_executors: Union[Dict[str, Executor], None] = None,
        full_name: Union[str, None] = None,
        key: Any = "",
    ):
        super().__init__(full_name=full_name, key=key)
        self.inner_executors = self._get_inner_executors(inner_executors)


class Experiment(Executor):
    def _detect_transformer(self) -> bool:
        return all(executor._is_transformer for executor in self.executors)

    def get_executor_ids(
        self, searched_classes: Union[type, Iterable[type], None] = None
    ) -> Dict[type, List[str]]:
        # странно
        if not searched_classes:
            return {}

        searched_classes = (
            searched_classes
            if isinstance(searched_classes, Iterable)
            else [searched_classes]
        )
        return {
            searched_class: [
                executor.id
                for executor in self.executors
                if isinstance(executor, searched_class)
            ]
            for searched_class in searched_classes
        }

    def __init__(
        self,
        executors: List[Executor],
        transformer: Union[bool, None] = None,
        full_name: Union[str, None] = None,
        key: Any = "",
    ):
        self.executors: List[Executor] = executors
        self.transformer: bool = (
            transformer if transformer is not None else self._detect_transformer()
        )
        full_name = str(full_name or f"Experiment({len(self.executors)})")
        super().__init__(full_name, key)

    # может быть удален?
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
        analyzer: Executor,
        full_name: Union[str, None] = None,
        key: Any = "",
    ):
        self.inner_executor: Executor = inner_executor
        self.n_iterations: int = n_iterations
        self.analyzer: Executor = analyzer
        super().__init__(full_name, key)

    def generate_params_hash(self) -> str:
        return f"{self.inner_executor.full_name} x {self.n_iterations}"

    def execute(self, data: ExperimentData) -> ExperimentData:
        for _ in range(self.n_iterations):
            data = self.analyzer.execute(self.inner_executor.execute(data))
        return data


class GroupExperiment(Executor):
    def generate_params_hash(self) -> str:
        return (
            f"GroupExperiment: {self.inner_executor._id.replace(ID_SPLIT_SYMBOL, '|')}"
        )

    def __init__(
        self,
        inner_executor: Executor,
        full_name: Union[str, None] = None,
        key: Any = "",
    ):
        self.inner_executor: Executor = inner_executor
        super().__init__(full_name, key)

    def _extract_result(self, data: ExperimentData) -> Dataset:
        return data.analysis_tables[self.inner_executor._id]

    def _insert_result(
        self, data: ExperimentData, result_list: List[Dataset]
    ) -> ExperimentData:
        result = result_list[0]
        for i in range(1, len(result_list)):
            result = result.append(result_list[i])
        data.set_value(
            ExperimentDataEnum.analysis_tables, self._id, self.full_name, result
        )
        return data

    def execute(self, data: ExperimentData) -> ExperimentData:
        result_list = []
        group_field = data.get_columns_by_roles(TempGroupingRole(), tmp_role=True)

        for group, group_data in data.groupby(group_field):
            temp_data = ExperimentData(group_data)
            temp_data = self.inner_executor.execute(temp_data)
            result_list.append(self._extract_result(temp_data))
        return self._insert_result(data, result_list)


class OnRoleExperiment(Experiment):
    def __init__(
        self,
        executors: List[Executor],
        role: ABCRole,
        transformer: Union[bool, None] = None,
        full_name: Union[str, None] = None,
        key: Any = "",
    ):
        self.role: ABCRole = role
        super().__init__(executors, transformer, full_name, key)

    def execute(self, data: ExperimentData) -> ExperimentData:
        for field in data.get_columns_by_roles(self.role):
            data.tmp_roles = {field: TempTargetRole()}
            data = super().execute(data)
            data.tmp_roles = {}
        return data
