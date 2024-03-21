from abc import ABC, abstractmethod
from typing import Callable

from hypex.experiment.base import Executor
from hypex.dataset.dataset import Dataset, ExperimentData
from hypex.dataset.roles import GroupingRole, TempTargetRole
from hypex.utils.hypex_typings import FieldKey
from hypex.stats.descriptive import Mean


class GroupComparator(ABC, Executor):
    def __init__(
        self,
        full_name: str = None,
        index: int = 0,
    ):
        super().__init__(full_name, index)

    @abstractmethod
    def _comparison_function(self, control_data, test_data):
        raise NotImplementedError

    def _compare(self, data: ExperimentData) -> Dict:
        group_field = data.data.get_columns_by_roles(GroupingRole)
        target_field = data.data.get_columns_by_roles(TempTargetRole, tmp_role=True)[0]
        grouping_data = list(data.groupby(self.group_field))
        return {
            grouping_data[i][0]: self._comparison_function(
                grouping_data[0][1][target_field],
                grouping_data[i][1][target_field],
            )
            for i in range(1, len(grouping_data))
        }

    def _set_value(self, data: ExperimentData, value: Dataset) -> ExperimentData:
        data.set_value("analysis_tables", self._id, self.get_full_name(), value)
        return data

    def _extract_dataset(self, compare_result: Dict, roles=None) -> Dataset:
        return Dataset(roles=roles).from_dict(compare_result)

    def execute(self, data: ExperimentData) -> ExperimentData:
        compare_result = self._compare(data)
        result_dataset = self._extract_dataset(compare_result)
        return self._set_value(data, result_dataset)


def GroupDifference(ComplexExecutor, GroupComparator):
    default_inner_executors: Dict[str, Executor] = {
        "mean": Mean(),
    }

    def _comparison_function(self, control_data, test_data) -> Dataset:
        mean_a = self._inner_executors["mean"].execute(control_data)
        mean_b = self._inner_executors["mean"].execute(test_data)
        return {
            "test mean": mean_a,
            "control mean": mean_b,
            "difference": mean_b - mean_a,
            "difference %": (mean_b / mean_a - 1) * 100,
        }
