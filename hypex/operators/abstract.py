from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union, Iterable

from hypex.dataset import (
    ABCRole,
    Dataset,
    ExperimentData,
    GroupingRole,
)
from hypex.executor import Calculator
from hypex.utils import (
    ComparisonNotSuitableFieldError,
    ExperimentDataEnum,
    FieldKeyTypes,
    NoColumnsError,
    SpaceEnum,
    AbstractMethodError,
)


class GroupOperator(Calculator):
    def __init__(
        self,
        grouping_role: Optional[ABCRole] = None,
        target_roles: Optional[List[ABCRole]] = None,
        space: SpaceEnum = SpaceEnum.auto,
        search_types: Union[type, List[type], None] = None,
        key: Any = "",
    ):
        self.grouping_role = grouping_role or GroupingRole()
        self.space = space
        self.__additional_mode = space == SpaceEnum.additional
        self._search_types = (
            search_types
            if isinstance(search_types, Iterable) or search_types is None
            else [search_types]
        )
        self.target_roles = target_roles or []
        super().__init__(key=key)

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

    def __group_field_searching(self, data: ExperimentData):
        group_field = []
        if self.space in [SpaceEnum.auto, SpaceEnum.data]:
            group_field = data.ds.search_columns(self.grouping_role)
        if (
            self.space in [SpaceEnum.auto, SpaceEnum.additional]
            and group_field == []
            and isinstance(data, ExperimentData)
        ):
            group_field = data.additional_fields.search_columns(self.grouping_role)
            self.__additional_mode = True
        if len(group_field) == 0:
            raise NoColumnsError(self.grouping_role)
        return group_field

    def __get_grouping_data(self, data: ExperimentData):
        group_field = self.__group_field_searching(data)
        target_fields = data.ds.search_columns(
            self.target_roles, search_types=self._search_types
        )
        if (
            not target_fields and data.ds.tmp_roles
        ):  # если колонка не подходит для теста, то тагет будет пустой, но если есть темп роли, то это нормальное поведение
            return data
        grouping_data = None
        if group_field[0] in data.groups:  # TODO: to recheck if this is a correct check
            grouping_data = list(data.groups[group_field[0]].items())
        return group_field, target_fields, grouping_data

    @staticmethod
    def __field_arg_universalization(
        field: Union[Sequence[FieldKeyTypes], FieldKeyTypes, None]
    ) -> List[FieldKeyTypes]:
        if not field:
            raise NoColumnsError(field)
        elif isinstance(field, FieldKeyTypes):
            return [field]
        return list(field)

    @classmethod
    def calc(
        cls,
        data: Dataset,
        group_field: Union[Sequence[FieldKeyTypes], FieldKeyTypes, None] = None,
        grouping_data: Optional[Dict[FieldKeyTypes, Dataset]] = None,
        target_fields: Optional[List[FieldKeyTypes]] = None,
        **kwargs,
    ) -> Dict:
        if len(target_fields) != 2:
            raise ValueError
        group_field = GroupOperator.__field_arg_universalization(group_field)

        if grouping_data is None:
            grouping_data = data.groupby(group_field)
        if len(grouping_data) > 1:
            grouping_data[0][1].tmp_roles = data.tmp_roles
        else:
            raise ComparisonNotSuitableFieldError(group_field)

        result = {}
        for group, group_data in grouping_data:
            control = group_data[target_fields[0]]
            test = group_data[target_fields[1]]
            result[group[0]] = cls._inner_function(
                data=control,
                test_data=test,
                **kwargs,
            )
        return result

    def _set_value(
        self, data: ExperimentData, value: Optional[Dict] = None, key: Any = None
    ) -> ExperimentData:
        data.set_value(
            ExperimentDataEnum.variables,
            self.id,
            str(self.__class__.__name__),
            value,
        )
        return data

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields, grouping_data = self.__get_grouping_data(data)
        compare_result = self.calc(
            data=data.ds,
            group_field=group_field,
            target_fields=target_fields,
            grouping_data=grouping_data,
        )
        return self._set_value(data, compare_result)
