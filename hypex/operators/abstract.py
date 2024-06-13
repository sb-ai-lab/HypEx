from typing import Any, Dict, Optional, Sequence, Union, List
from hypex.dataset import Dataset, ExperimentData, ABCRole
from hypex.executor import GroupCalculator
from hypex.utils.enums import ExperimentDataEnum, SpaceEnum
from hypex.utils.errors import ComparisonNotSuitableFieldError
from hypex.utils.typings import FieldKeyTypes


class GroupOperator(GroupCalculator):
    def __init__(
        self,
        grouping_role: Optional[ABCRole] = None,
        target_roles: Optional[ABCRole] = None,
        space: SpaceEnum = SpaceEnum.auto,
        search_types: Union[type, List[type], None] = None,
        key: Any = "",
    ):
        self.target_roles = target_roles
        super().__init__(grouping_role=grouping_role, space=space, search_types=search_types, key=key)

    def __get_grouping_data(self, data: ExperimentData):
        group_field = self.__group_field_searching(data)
        target_fields = data.ds.search_columns(
            self.target_roles, tmp_role=True, search_types=self._search_types
        )
        if (
            not target_fields and data.ds.tmp_roles
        ):  # если колонка не подходит для теста, то тагет будет пустой, но если есть темп роли, то это нормальное поведение
            return data
        grouping_data = None
        if group_field[0] in data.groups:  # TODO: to recheck if this is a correct check
            grouping_data = list(data.groups[group_field[0]].items())
        return group_field, target_fields, grouping_data

    @classmethod
    def calc(
        cls,
        data: Dataset,
        group_field: Union[Sequence[FieldKeyTypes], FieldKeyTypes, None] = None,
        target_fields: Optional[FieldKeyTypes] = None,
        grouping_data: Optional[Dict[FieldKeyTypes, Dataset]] = None,
        **kwargs,
    ) -> Dict:
        if len(target_fields) != 2: 
            raise ValueError
        group_field = GroupCalculator.__field_arg_universalization(group_field)

        if grouping_data is None:
            grouping_data = data.groupby(group_field)
        if len(grouping_data) > 1:
            grouping_data[0][1].tmp_roles = data.tmp_roles
        else:
            raise ComparisonNotSuitableFieldError(group_field)

        result = {}
        for group, group_data in grouping_data: 
            result[group] = cls._inner_function(
                    data=group_data[target_fields[0]],
                    test_data=group_data[target_fields[1]],
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