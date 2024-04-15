from abc import abstractmethod
from typing import Dict, Any, Optional

from hypex.dataset.dataset import Dataset, ExperimentData
from hypex.dataset.roles import GroupingRole, TempTargetRole, ABCRole, StatisticRole
from hypex.experiments.base import Executor, ComplexExecutor
from hypex.operators.binary import MetricDelta
from hypex.stats.descriptive import Mean, Size
from hypex.utils.enums import ExperimentDataEnum, SpaceEnum, BackendsEnum
from hypex.utils.errors import NoColumnsError, ComparisonNotSuitableFieldError
from hypex.utils.typings import FromDictType


class GroupComparator(ComplexExecutor):
    def __init__(
        self,
        grouping_role: Optional[ABCRole] = None,
        space: SpaceEnum = SpaceEnum.auto,
        inner_executors: Optional[Dict[str, Executor]] = None,
        full_name: Optional[str] = None,
        key: Any = "",
    ):
        self.grouping_role = grouping_role or GroupingRole()
        self.space = space
        self.__additional_mode = space == SpaceEnum.additional
        super().__init__(inner_executors=inner_executors, full_name=full_name, key=key)

    def _local_extract_dataset(
        self, compare_result: Dict[Any, Any], roles: Dict[Any, ABCRole]
    ) -> Dataset:
        return self._extract_dataset(compare_result, roles)

    def _one_stat_calculation(self, data, stat: Executor):
        pass

    @abstractmethod
    def _comparison_function(self, control_data, test_data) -> Dict[str, Any]:
        raise NotImplementedError

    def __group_field_searching(self, data: ExperimentData):
        group_field = []
        if self.space in [SpaceEnum.auto, SpaceEnum.data]:
            group_field = data.get_columns_by_roles(self.grouping_role)
        if (
            self.space in [SpaceEnum.auto, SpaceEnum.additional]
            and group_field == []
            and isinstance(data, ExperimentData)
        ):
            group_field = data.additional_fields.get_columns_by_roles(
                self.grouping_role
            )
            self.__additional_mode = True
        if len(group_field) == 0:
            raise NoColumnsError(self.grouping_role)
        return group_field

    def __get_grouping_data(self, data: ExperimentData, group_field):
        if self.__additional_mode:
            t_groups = list(data.additional_fields.groupby(group_field))
            result = [(group, data.loc[subdata.index]) for (group, subdata) in t_groups]
        else:
            result = list(data.groupby(group_field))

        result = [
            (group[0] if len(group) == 1 else group, subdata)
            for (group, subdata) in result
        ]
        return result

    def calc(self, data: Dataset) -> Dict:
        target_field = None
        group_field = self.__group_field_searching(data)
        meta_name = group_field[0] if len(group_field) == 1 else group_field
        group_name = (
            str(data.id_name_mapping.get(meta_name, meta_name))
            if (self.__additional_mode and isinstance(data, ExperimentData))
            else str(meta_name)
        )[0]
        target_field = data.get_columns_by_roles(TempTargetRole(), tmp_role=True)[0]
        self.key = f"{target_field}[{group_name}]" + self.key
        grouping_data = self.__get_grouping_data(data, group_field)
        if len(grouping_data) > 1:
            grouping_data[0][1].tmp_roles = data.tmp_roles
        else:
            raise ComparisonNotSuitableFieldError(group_field)

        result = {}
        if target_field:
            for i in range(1, len(grouping_data)):
                grouping_data[i][1].tmp_roles = data.tmp_roles
                result[grouping_data[i][0]] = self._comparison_function(
                    grouping_data[0][1][target_field],
                    grouping_data[i][1][target_field],
                )
        else:
            for i in range(1, len(grouping_data)):
                result[grouping_data[i][0]] = self._comparison_function(
                    grouping_data[0][1], grouping_data[i][1]
                )
        return result

    def _set_value(
        self, data: ExperimentData, value: Optional[Dataset, None] = None, key: Any = None
    ) -> ExperimentData:
        data.set_value(
            ExperimentDataEnum.analysis_tables, self.id, str(self.full_name), value
        )
        return data

    @staticmethod
    def _extract_dataset(
        compare_result: FromDictType, roles: Dict[Any, ABCRole]
    ) -> Dataset:
        return Dataset.from_dict(compare_result, roles, BackendsEnum.pandas)

    def execute(self, data: ExperimentData) -> ExperimentData:
        compare_result = self.calc(data)
        result_dataset = self._local_extract_dataset(
            compare_result, {key: StatisticRole() for key, _ in compare_result.items()}
        )
        return self._set_value(data, result_dataset)


class GroupDifference(GroupComparator):
    default_inner_executors: Dict[str, Executor] = {
        "mean": Mean(),
    }

    def _comparison_function(self, control_data, test_data) -> Dict[str, Any]:
        target_field = control_data.get_columns_by_roles(
            TempTargetRole(), tmp_role=True
        )[0]
        ed_control = self.inner_executors["mean"].calc(control_data)
        ed_test = self.inner_executors["mean"].calc(test_data)

        mean_a = ed_control.iloc[0]
        mean_b = ed_test.iloc[0]

        return {
            f"{target_field} control mean": mean_a,
            f"{target_field} test mean": mean_b,
            f"{target_field} difference": mean_b - mean_a,
            f"{target_field} difference %": (mean_b / mean_a - 1) * 100,
        }


class GroupSizes(GroupComparator):
    default_inner_executors: Dict[str, Executor] = {
        "size": Size(),
    }

    def _comparison_function(self, control_data, test_data) -> Dict[str, Any]:
        size_a = self.inner_executors["size"].calc(control_data)
        size_b = self.inner_executors["size"].calc(test_data)

        return {
            "control size": size_a,
            "test size": size_b,
            "control size %": (size_a / (size_a + size_b)) * 100,
            "test size %": (size_b / (size_a + size_b)) * 100,
        }


class GroupATE(GroupComparator):
    default_inner_executors: Dict[str, Executor] = {
        "delta": MetricDelta(),
        "mean": Mean(),
        "size": Size(),
    }

    def _comparison_function(self, control_data, test_data) -> Dict[str, Any]:
        target_field = control_data.get_columns_by_roles(
            TempTargetRole(), tmp_role=True
        )[0]
        size_a = self.inner_executors["size"].calc(control_data)
        size_b = self.inner_executors["size"].calc(test_data)
        control_mean = self.inner_executors["mean"].calc(control_data)
        test_mean = self.inner_executors["mean"].calc(test_data)

        ate = (size_a / (size_a + size_b)) * control_mean + (
            size_b / (size_a + size_b)
        ) * test_mean

        return {f"{target_field} ATE": ate.iloc[0]}
