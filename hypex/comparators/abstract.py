from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from hypex.dataset import (
    ABCRole,
    Dataset,
    ExperimentData,
    GroupingRole,
    StatisticRole,
    TempTargetRole,
)
from hypex.executor import Calculator
from hypex.utils import (
    BackendsEnum,
    ComparisonNotSuitableFieldError,
    ExperimentDataEnum,
    FieldKeyTypes,
    FromDictTypes,
    NoColumnsError,
    SpaceEnum,
)
from hypex.utils.errors import AbstractMethodError


class GroupComparator(Calculator):
    def __init__(
            self,
            grouping_role: Optional[ABCRole] = None,
            space: SpaceEnum = SpaceEnum.auto,
            key: Any = "",
    ):
        self.grouping_role = grouping_role or GroupingRole()
        self.space = space
        self.__additional_mode = space == SpaceEnum.additional
        super().__init__(key=key)

    def _local_extract_dataset(
            self, compare_result: Dict[Any, Any], roles: Dict[Any, ABCRole]
    ) -> Dataset:
        return self._extract_dataset(compare_result, roles)

    @staticmethod
    @abstractmethod
    def _inner_function(data: Dataset, test_data: Dataset, **kwargs) -> Any:
        raise AbstractMethodError

    def __group_field_searching(self, data: ExperimentData):
        group_field = []
        if self.space in [SpaceEnum.auto, SpaceEnum.data]:
            group_field = data.ds.search_columns(self.grouping_role)
        if (
                self.space in [SpaceEnum.auto, SpaceEnum.additional]
                and group_field == []
                and isinstance(data, ExperimentData)
        ):
            group_field = data.additional_fields.search_columns(
                self.grouping_role
            )
            self.__additional_mode = True
        if len(group_field) == 0:
            raise NoColumnsError(self.grouping_role)
        return group_field

    def __get_grouping_data(self, data: ExperimentData, group_field):
        if self.__additional_mode:
            t_groups = list(data.additional_fields.groupby(group_field))
            result = [
                (group, data.ds.loc[subdata.index]) for (group, subdata) in t_groups
            ]
        else:
            result = list(data.ds.groupby(group_field))

        result = [
            (group[0] if len(group) == 1 else group, subdata)
            for (group, subdata) in result
        ]
        return result

    @staticmethod
    def __field_arg_universalization(
            field: Union[Sequence[FieldKeyTypes], FieldKeyTypes, None]
    ) -> List[FieldKeyTypes]:
        if not field:
            raise NoColumnsError(field)
        elif isinstance(field, FieldKeyTypes):
            return [field]
        return list(field)

    @staticmethod
    @abstractmethod
    def _to_dataset(data: Any, **kwargs) -> Dataset:
        raise AbstractMethodError

    @classmethod
    def calc(
            cls,
            data: Dataset,
            group_field: Union[Sequence[FieldKeyTypes], FieldKeyTypes, None] = None,
            target_field: Optional[FieldKeyTypes] = None,
            grouping_data: Optional[Dict[FieldKeyTypes, Dataset]] = None,
            **kwargs,
    ) -> Dict:
        group_field = GroupComparator.__field_arg_universalization(group_field)

        if grouping_data is None:
            grouping_data = data.groupby(group_field)
        if len(grouping_data) > 1:
            grouping_data[0][1].tmp_roles = data.tmp_roles
        else:
            raise ComparisonNotSuitableFieldError(group_field)

        result = {}
        if target_field:
            for i in range(1, len(grouping_data)):
                result_key = (
                    grouping_data[i][0]
                    if len(grouping_data[i][0]) > 1
                    else grouping_data[i][0][0]
                )
                grouping_data[i][1].tmp_roles = data.tmp_roles
                result[result_key] = cls._to_dataset(cls._inner_function(
                    grouping_data[0][1][target_field],
                    grouping_data[i][1][target_field],
                    **kwargs,
                ))
        else:
            for i in range(1, len(grouping_data)):
                result_key = (
                    grouping_data[i][0]
                    if len(grouping_data[i][0]) > 1
                    else grouping_data[i][0][0]
                )
                result[result_key] = cls._to_dataset(cls._inner_function(
                    grouping_data[0][1],
                    grouping_data[i][1],
                    **kwargs,
                ))
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
            compare_result: FromDictTypes, roles: Dict[Any, ABCRole]
    ) -> Dataset:
        return Dataset.from_dict(compare_result, roles, BackendsEnum.pandas)

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field = self.__group_field_searching(data)
        target_fields = data.ds.get_columns_by_roles(TempTargetRole(), tmp_role=True)
        if group_field in data.groups:  # TODO: to recheck if this is a correct check
            grouping_data = list(data.groups[group_field].items())
        else:
            grouping_data = None
        compare_result = self.calc(
            data=data.ds,
            group_field=group_field,
            target_field=target_fields,
            grouping_data=grouping_data,
            comparison_function=self._inner_function,
        )
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
            full_name: Union[str, None] = None,
            key: Any = "",
    ):
        super().__init__(grouping_role, space, full_name, key)
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

        return super()._extract_dataset(
            result_stats,
            roles={
                f: StatisticRole() for f in ["group", "statistic", "p-value", "pass"]
            },
        )
