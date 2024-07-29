from copy import deepcopy
from typing import Iterable, Dict, Union, Any, List, Optional, Sequence

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
        executors: Sequence[Executor],
        transformer: Optional[bool] = None,
        key: Any = "",
    ):
        self.executors: Sequence[Executor] = executors
        self.transformer: bool = (
            transformer if transformer is not None else self._detect_transformer()
        )
        super().__init__(key)

    def set_params(
        self, params: Union[Dict[str, Any], Dict[type, Dict[str, Any]]]
    ) -> None:
        if isinstance(list(params)[0], str):
            super().set_params(params)
        elif isinstance(list(params)[0], type):
            for executor in self.executors:
                executor.set_params(params)
        else:
            raise ValueError(
                "params must be a dict of str to dict or a dict of class to dict"
            )

    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        return data.set_value(ExperimentDataEnum.analysis_tables, self.id, value)

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
        role: Union[ABCRole, Sequence[ABCRole]],
        transformer: Optional[bool] = None,
        key: Any = "",
    ):
        self.role: List[ABCRole] = [role] if isinstance(role, ABCRole) else list(role)
        super().__init__(executors, transformer, key)

    def execute(self, data: ExperimentData) -> ExperimentData:
        for field in data.ds.search_columns(self.role):
            data.ds.tmp_roles = {field: TempTargetRole()}
            data = super().execute(data)
            data.ds.tmp_roles = {}
        return data
