from abc import abstractmethod
from typing import Dict, Union, Any

from hypex.utils.typings import FromDictType

from hypex.dataset.dataset import Dataset, ExperimentData
from hypex.dataset.roles import GroupingRole, TempTargetRole, ABCRole
from hypex.experiment.experiment import Executor, ComplexExecutor
from hypex.stats.descriptive import Mean, Size
from hypex.utils.enums import ExperimentDataEnum

# TODO: replace space om Enum
class GroupComparator(ComplexExecutor):
    def __init__(
        self,
        grouping_role: Union[ABCRole, None] = None,
        space: str = "auto",
        inner_executors: Union[Dict[str, Executor], None] = None,
        full_name: Union[str, None] = None,
        key: Any = 0,
    ):
        self.grouping_role = grouping_role or GroupingRole()
        self.space = space
        self.__additional_mode = space == "additional"
        super().__init__(inner_executors=inner_executors, full_name=full_name, key=key)

    def _local_extract_dataset(self, compare_result: Dict[Any, Any]) -> Dataset:
        return self._extract_dataset(compare_result)

    @abstractmethod
    def _comparison_function(self, control_data, test_data):
        raise NotImplementedError

    def __group_field_searching(self, data: ExperimentData):
        group_field = []
        if self.space in ["auto", "data"]:
            group_field = data.get_columns_by_roles(self.grouping_role)
        if self.space in ["auto", "additional"]:
            group_field = data.additional_fields.get_columns_by_roles(self.grouping_role)
            self.__additional_mode = True
        if len(group_field) == 0:
            raise ValueError(f"No columns found by role {self.grouping_role}")
        return group_field

    def __get_grouping_data(self, data: ExperimentData, group_field):
        if self.__additional_mode:
            t_groups = list(data.additional_fields.groupby(group_field))
            result = [(group, data.loc[subdata.index]) for (group, subdata) in t_groups]
        else:
            result = list(data.groupby(group_field))
        
        result = [(group[0] if len(group) == 1 else group, subdata) for (group, subdata) in result]
        return result


    def _compare(self, data: ExperimentData) -> Dict:
        group_field = self.__group_field_searching(data)
        group_name = str(group_field) if not self.__additional_mode else str(data._id_name_mapping.get(group_field[0], group_field))
        target_field = data.get_columns_by_roles(TempTargetRole(), tmp_role=True)[0]
        self.key = f"{target_field}[{group_name}]"
        grouping_data = self.__get_grouping_data(data, group_field)
        if len(grouping_data) > 1:
            grouping_data[0][1].tmp_roles = data.tmp_roles
        else:
            raise ValueError(f"Group field {group_field} is not suitable for comparison")

        result = {}
        for i in range(1, len(grouping_data)):
            grouping_data[i][1].tmp_roles = data.tmp_roles
            result[grouping_data[i][0]] = self._comparison_function(
                grouping_data[0][1][target_field],
                grouping_data[i][1][target_field],
            )
        return result

    def _set_value(self, data: ExperimentData, value: Dataset) -> ExperimentData:
        data.set_value(
            ExperimentDataEnum.analysis_tables, self.id, self.full_name, value
        )
        return data

    def _extract_dataset(
        self, compare_result: FromDictType, roles: Union[ABCRole, None] = None
    ) -> Dataset:
    #TODO: change
        return Dataset(roles=roles).from_dict(compare_result)

    def execute(self, data: ExperimentData) -> ExperimentData:
        compare_result = self._compare(data)
        result_dataset = self._local_extract_dataset(compare_result)
        return self._set_value(data, result_dataset)


class GroupDifference(GroupComparator):
    default_inner_executors: Dict[str, Executor] = {
        "mean": Mean(),
    }

    def _comparison_function(self, control_data, test_data) -> Dataset:
        target_field = control_data.get_columns_by_roles(
            TempTargetRole(), tmp_role=True
        )[0]
        mean_a = self.inner_executors["mean"].calc(control_data).iloc[0]
        mean_b = self.inner_executors["mean"].calc(test_data).iloc[0]

        return {
            f"{target_field} control mean": mean_a,
            f"{target_field} test mean": mean_b,
            f"{target_field} difference": mean_b - mean_a,
            f"{target_field} difference %": (mean_b / mean_a - 1) * 100,
        }


class GroupSizes(GroupComparator):
    default_inner_executors: Dict[str, Executor] = {
        "mean": Size(),
    }

    def _comparison_function(self, control_data, test_data) -> Dataset:
        size_a = self.inner_executors["size"].execute(control_data)
        size_b = self.inner_executors["size"].execute(test_data)

        return {
            "control size": size_a,
            "test size": size_b,
            "control size %": (size_a / (size_a + size_b)) * 100,
            "test size %": (size_b / (size_a + size_b)) * 100,
        }
