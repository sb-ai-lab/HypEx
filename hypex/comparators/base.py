from abc import abstractmethod, ABC
from typing import Optional, Dict, Any, List, Union

from hypex.dataset.dataset import Dataset, ExperimentData
from hypex.dataset.roles import ABCRole, GroupingRole, TempTargetRole, StatisticRole
from hypex.experiments.base import ComplexExecutor, Executor
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
        target_field = data.get_columns_by_roles(TempTargetRole(), tmp_role=True)
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
        self, data: ExperimentData, value: Optional[Dataset] = None, key: Any = None
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


class StatHypothesisTestingWithScipy(GroupComparator, ABC):
    def __init__(
        self,
        grouping_role: Union[ABCRole, None] = None,
        space: SpaceEnum = SpaceEnum.auto,
        reliability: float = 0.05,
        inner_executors: Union[Dict[str, Executor], None] = None,
        full_name: Union[str, None] = None,
        key: Any = "",
    ):
        super().__init__(grouping_role, space, inner_executors, full_name, key)
        self.reliability = reliability

    # excessive override
    def _local_extract_dataset(
        self, compare_result: Dict[Any, Any], roles=None
    ) -> Dataset:
        # stats type
        result_stats: List[Dict[str, Any]] = [
            {
                "group": group,
                "statistic": stats.statistic,
                "p-value": stats.pvalue,
                "pass": stats.pvalue < self.reliability,
            }
            for group, stats in compare_result.items()
        ]
        # mypy does not see an heir
        # return super()._extract_dataset(
        #     result_stats,
        #     roles={StatisticRole(): ["group", "statistic", "p-value", "pass"]}
        # )

        return super()._extract_dataset(
            result_stats,
            roles={
                f: StatisticRole() for f in ["group", "statistic", "p-value", "pass"]
            },
        )
