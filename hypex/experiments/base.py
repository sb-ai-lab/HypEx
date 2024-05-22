from copy import deepcopy
from typing import Iterable, Dict, Union, Any, List, Optional

from hypex.dataset import (
    ExperimentData,
    Dataset,
    TempGroupingRole,
    TempTargetRole,
    ABCRole,
    GroupingRole,
    TreatmentRole,
    TempTreatmentRole,
)
from hypex.executor import Executor
from hypex.utils import ID_SPLIT_SYMBOL, ExperimentDataEnum

from tqdm.auto import tqdm


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
        full_name: Optional[str] = None,
        key: Any = "",
    ):
        self.executors: List[Executor] = executors
        self.transformer: bool = (
            transformer if transformer is not None else self._detect_transformer()
        )
        full_name = str(full_name or f"Experiment({len(self.executors)})")
        super().__init__(full_name, key)

    def execute(self, data: ExperimentData) -> ExperimentData:
        experiment_data = deepcopy(data) if self.transformer else data
        for executor in self.executors:
            executor.key = self.key
            executor.random_state = self.random_state
            experiment_data = executor.execute(experiment_data)
        return experiment_data


class CycledExperiment(Executor):
    def __init__(
        self,
        inner_executor: Executor,
        n_iterations: int,
        analyzer: Executor,
        full_name: Optional[str] = None,
        key: Any = "",
    ):
        self.inner_executor: Executor = inner_executor
        self.n_iterations: int = n_iterations
        self.analyzer: Executor = analyzer
        super().__init__(full_name, key)

    def generate_params_hash(self) -> str:
        return f"{self.inner_executor.full_name} x {self.n_iterations}"

    def execute(self, data: ExperimentData) -> ExperimentData:
        for i in tqdm(range(self.n_iterations)):
            self.analyzer.key = f"{i}"
            self.inner_executor.key = f"{i}"
            self.inner_executor.random_state = i
            data = self.analyzer.execute(self.inner_executor.execute(data))
            column = data.additional_fields.get_columns_by_roles(TreatmentRole())[0]
            data.additional_fields.roles[column] = TempTreatmentRole()
        return data


class GroupExperiment(Executor):
    def generate_params_hash(self) -> str:
        return (
            f"GroupExperiment: {self.inner_executor._id.replace(ID_SPLIT_SYMBOL, '|')}"
        )

    def __init__(
        self,
        inner_executor: Executor,
        full_name: Optional[str] = None,
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
            ExperimentDataEnum.analysis_tables, self._id, str(self.full_name), result
        )
        return data

    def execute(self, data: ExperimentData) -> ExperimentData:
        result_list = []
        group_field = data.ds.get_columns_by_roles(TempGroupingRole(), tmp_role=True)

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
        full_name: Optional[str] = None,
        key: Any = "",
    ):
        self.role: ABCRole = role
        super().__init__(executors, transformer, full_name, key)

    def execute(self, data: ExperimentData) -> ExperimentData:
        for field in data.ds.get_columns_by_roles(self.role):
            data.ds.tmp_roles = {field: TempTargetRole()}
            data = super().execute(data)
            data.ds.tmp_roles = {}
        return data
