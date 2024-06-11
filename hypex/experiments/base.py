from copy import deepcopy
from typing import Iterable, Dict, Union, Any, List, Optional

from hypex.dataset import (
    ExperimentData,
    ABCRole,
    TempTargetRole,
)
from hypex.executor import Executor
from hypex.utils import ExperimentDataEnum


class Experiment(Executor):
    def _detect_transformer(self) -> bool:
        return all(executor._is_transformer for executor in self.executors)

    def get_executor_ids(
        self, searched_classes: Union[type, Iterable[type], None] = None
    ) -> Dict[type, List[str]]:
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
        transformer: Optional[bool] = None,
        key: Any = "",
    ):
        self.executors: List[Executor] = executors
        self.transformer: bool = (
            transformer if transformer is not None else self._detect_transformer()
        )
        super().__init__(key)

    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.analysis_tables, self.id, self.__class__.__name__, value
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        experiment_data = deepcopy(data) if self.transformer else data
        for executor in self.executors:
            executor.key = self.key
            experiment_data = executor.execute(experiment_data)
        return experiment_data


class OnRoleExperiment(Experiment):
    def __init__(
        self,
        executors: List[Executor],
        role: ABCRole,
        transformer: Optional[bool] = None,
        key: Any = "",
    ):
        self.role: ABCRole = role
        super().__init__(executors, transformer, key)

    def execute(self, data: ExperimentData) -> ExperimentData:
        for field in data.ds.search_columns(self.role):
            data.ds.tmp_roles = {field: TempTargetRole()}
            data = super().execute(data)
            data.ds.tmp_roles = {}
        return data
