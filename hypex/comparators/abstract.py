from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from hypex.dataset import (
    ABCRole,
    Dataset,
    ExperimentData,
    StatisticRole,
    TempTargetRole,
    MatchingRole,
)
from hypex.executor import GroupCalculator
from hypex.utils import (
    BackendsEnum,
    ExperimentDataEnum,
    FieldKeyTypes,
    FromDictTypes,
    SpaceEnum,
    AbstractMethodError,
)


class GroupComparator(GroupCalculator):
    def __init__(
        self,
        grouping_role: Optional[ABCRole] = None,
        space: SpaceEnum = SpaceEnum.auto,
        search_types: Union[type, List[type], None] = None,
        key: Any = "",
    ):
        self.__additional_mode = space == SpaceEnum.additional
        super().__init__(
            grouping_role=grouping_role, search_types=search_types, space=space, key=key
        )

    def _local_extract_dataset(
        self, compare_result: Dict[Any, Any], roles: Dict[Any, ABCRole]
    ) -> Dataset:
        return self._extract_dataset(compare_result, roles)

    @staticmethod
    def _check_test_data(test_data: Optional[Dataset] = None) -> Dataset:
        if test_data is None:
            raise ValueError("test_data is needed for comparison")
        return test_data

    @classmethod
    @abstractmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Any:
        raise AbstractMethodError

    def _get_fields(self, data: ExperimentData):
        group_field = self._field_searching(data, self.grouping_role)
        target_fields = data.ds.search_columns(
            TempTargetRole(), tmp_role=True, search_types=self._search_types
        )
        return group_field, target_fields

    @staticmethod
    def _to_dataset(data: Any, **kwargs) -> Dataset:
        return data

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_fields: Optional[List[FieldKeyTypes]] = None,
        old_data: Optional[Dataset] = None,
        **kwargs,
    ) -> Dict:
        result = {}
        for i in range(1, len(grouping_data)):
            result_key = (
                grouping_data[i][0]
                if len(grouping_data[i][0]) > 1
                else grouping_data[i][0][0]
            )
            if target_fields:
                grouping_data[i][1].tmp_roles = old_data.tmp_roles
                result[result_key] = cls._to_dataset(
                    cls._inner_function(
                        data=grouping_data[0][1][target_fields],
                        test_data=grouping_data[i][1][target_fields],
                        **kwargs,
                    )
                )
            else:
                result[result_key] = cls._to_dataset(
                    cls._inner_function(
                        grouping_data[0][1],
                        grouping_data[i][1],
                        **kwargs,
                    )
                )
        return result

    def _set_value(
        self, data: ExperimentData, value: Optional[Dataset] = None, key: Any = None
    ) -> ExperimentData:
        data.set_value(
            ExperimentDataEnum.analysis_tables,
            self.id,
            str(self.__class__.__name__),
            value,
        )
        return data

    @staticmethod
    def _extract_dataset(
        compare_result: FromDictTypes, roles: Dict[Any, ABCRole]
    ) -> Dataset:
        # TODO: криво. Надо переделать
        if isinstance(list(compare_result.values())[0], Dataset):
            cr_list_v: List[Dataset] = list(compare_result.values())
            result = cr_list_v[0]
            if len(cr_list_v) > 1:
                result = result.append(cr_list_v[1:])
            result.index = list(compare_result.keys())
            return result
        return Dataset.from_dict(compare_result, roles, BackendsEnum.pandas)

    # TODO выделить в отдельную функцию с кваргами (нужно для альфы)

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields, grouping_data = self._get_grouping_data(data)
        compare_result = self.calc(
            data=data.ds,
            group_field=group_field,
            target_fields=target_fields,
            grouping_data=grouping_data,
        )
        result_dataset = self._local_extract_dataset(
            compare_result, {key: StatisticRole() for key in compare_result}
        )
        return self._set_value(data, result_dataset)


class StatHypothesisTesting(GroupComparator, ABC):
    def __init__(
        self,
        grouping_role: Union[ABCRole, None] = None,
        space: SpaceEnum = SpaceEnum.auto,
        search_types: Union[type, List[type], None] = None,
        reliability: float = 0.05,
        key: Any = "",
    ):
        super().__init__(grouping_role, space, search_types, key)
        self.reliability = reliability


class MatchingComparator(GroupComparator):
    def __init__(
        self,
        grouping_role: Union[ABCRole, None] = None,
        space: SpaceEnum = SpaceEnum.auto,
        search_types: Union[type, List[type], None] = None,
        index_role: Union[ABCRole, None] = None,
        key: Any = "",
    ):
        super().__init__(grouping_role, space, search_types, key)
        self.index_role = index_role or MatchingRole()

    def _set_value(
        self,
        data: ExperimentData,
        value: Optional[Union[float, int, bool, str]] = None,
        key: Any = None,
    ) -> ExperimentData:
        data.set_value(
            ExperimentDataEnum.value,
            self.id,
            str(self.__class__.__name__),
            value,
        )
        return data

    @staticmethod
    def _inner_function(
        data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Any:
        raise AbstractMethodError

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields, grouping_data = self._get_grouping_data(data)
        if grouping_data:
            index_column = data.additional_fields.search_columns(self.index_role)
            grouping_data = [
                (
                    group,
                    (
                        group_data.iloc[index_column]
                        if group in index_column
                        else group_data
                    ),
                )
                for group, group_data in grouping_data
            ]
        att = data.variables.get("ATT", None)
        atc = data.variables.get("ATT", None)
        compare_result = self.calc(
            data=data.ds,
            group_field=group_field,
            target_field=target_fields,
            grouping_data=grouping_data,
            comparison_function=self._inner_function,
            att=att,
            atc=atc,
        )
        return self._set_value(
            data, list(compare_result.values())[0], key=list(compare_result.keys())[0]
        )
