from copy import deepcopy
from typing import Iterable, Dict, Union, Any, List, Optional

from tqdm.auto import tqdm

from hypex.dataset import (
    ExperimentData,
    Dataset,
    TempGroupingRole,
    TempTargetRole,
    ABCRole,
    GroupingRole,
)
from hypex.executor import Executor
from hypex.utils import ID_SPLIT_SYMBOL, ExperimentDataEnum


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

    def execute(self, data: ExperimentData) -> ExperimentData:
        experiment_data = deepcopy(data) if self.transformer else data
        for executor in self.executors:
            executor.key = self.key
            executor.random_state = self.random_state
            experiment_data = executor.execute(experiment_data)
        return experiment_data


# TODO: Reporter cycle import
class CycledExperiment(Executor):
    def __init__(
        self,
        inner_executor: Executor,
        n_iterations: int,
        reporter,
        key: Any = "",
    ):
        self.inner_executor: Executor = inner_executor
        self.n_iterations: int = n_iterations
        self.reporter = reporter
        super().__init__(key)

    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.analysis_tables, self.id, self.__class__.__name__, value
        )

    def generate_params_hash(self) -> str:
        return f"{self.inner_executor.__class__.__name__} x {self.n_iterations}"

    def execute(self, data: ExperimentData) -> ExperimentData:
        result: List[Dataset] = []
        for i in tqdm(range(self.n_iterations)):
            t_data = ExperimentData(data.ds)
            self.reporter.key = f"{i}"
            t_data = self.inner_executor.execute(t_data)
            result.append(self.reporter.report(t_data))
        result = result[0].append(result[1:], range(self.n_iterations))
        return self._set_value(data, result)


class GroupExperiment(Executor):
    def generate_params_hash(self) -> str:
        return (
            f"GroupExperiment: {self.inner_executor._id.replace(ID_SPLIT_SYMBOL, '|')}"
        )

    def __init__(
        self,
        inner_executor: Executor,
        key: Any = "",
    ):
        self.inner_executor: Executor = inner_executor
        super().__init__(key)

    def _extract_result(self, data: ExperimentData) -> Dataset:
        return data.analysis_tables[self.inner_executor._id]

    def _insert_result(
        self, data: ExperimentData, result_list: List[Dataset]
    ) -> ExperimentData:
        result = result_list[0]
        for i in range(1, len(result_list)):
            result = result.append(result_list[i])
        data.set_value(
            ExperimentDataEnum.analysis_tables,
            self._id,
            str(self.__class__.__name__),
            result,
        )
        return data

    def execute(self, data: ExperimentData) -> ExperimentData:
        result_list = []
        group_field = data.ds.search_columns(TempGroupingRole(), tmp_role=True)

        for group, group_data in data.ds.groupby(group_field):
            temp_data = ExperimentData(group_data)
            temp_data = self.inner_executor.execute(temp_data)
            temp_data = temp_data.ds.add_column(
                [group] * len(temp_data.ds),
                role={f"group({self.id_for_name})": GroupingRole()},
            )
            result_list.append(self._extract_result(temp_data))
        return self._insert_result(data, result_list)


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
