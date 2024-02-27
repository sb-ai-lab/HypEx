from abc import ABC, abstractmethod
from typing import Callable

from hypex.experiment.base import Executor
from hypex.dataset.dataset import ExperimentData
from hypex.dataset.roles import GroupingRole
from hypex.utils.hypex_typings import FieldKey

# TODO: to Experiment
class Comparator(Executor):
    def __init__(
        self,
        target_field: FieldKey,
        stat: StatDescriptive,
        full_name: str = None,
    ):
        self.target_field = target_field
        self.comparison_function = comparison_function
        super().__init__(full_name)

    def _compare(self, data: ExperimentData) -> bool:
        group_field = data.get_columns_by_roles(GroupingRole)[0]
        grouping_data = list(data.groupby(self.group_field))
        return {
            grouping_data[i][0]: self.comparison_function(
                grouping_data[0][1][self.target_field],
                grouping_data[i][1][self.target_field],
            )
            for i in range(1, len(grouping_data))
        }

    def _set_value(self, data: ExperimentData, value: Dataset) -> ExperimentData:
        data.set_value("analysis_tables", self._id, self.get_full_name(), value)
        return data

    def _extract_dataset(self, compare_result: Dict) -> Dataset:
        # TODO: not implemented
        return Dataset([compare_result])

    def execute(self, data: ExperimentData) -> ExperimentData:
        compare_result = self._compare(data)
        result_dataset = self._extract_dataset(compare_result)
        return self._set_value(data, result_dataset)


class ComparatorInner(ABC, Comparator):
    @abstractmethod
    def _comparison_function(self, control_data, test_data) -> ExperimentData:
        raise NotImplementedError

    def __init__(
        self,
        target_field: FieldKey,
        full_name: str = None,
    ):
        super().__init__(target_field, self._comparison_function, full_name)
