from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from hypex.dataset import (
    Dataset,
    ExperimentData, TargetRole, GroupingRole, ABCRole,
)
from hypex.executor import Calculator
from hypex.utils import (
    ExperimentDataEnum,
    AbstractMethodError,
    SpaceEnum,
)


class GroupOperator(Calculator):

    def __init__(
        self,
        grouping_role: Optional[ABCRole] = None,
        target_roles: Union[ABCRole, List[ABCRole], None] = None,
        key: Any = "",
    ):
        super().__init__(key=key)
        self.target_roles = target_roles or TargetRole()
        self.grouping_role = grouping_role or GroupingRole()

    @property
    def search_types(self):
        return None

    @classmethod
    @abstractmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Any:
        raise AbstractMethodError

    def _get_fields(self, data: ExperimentData):
        group_field = self._field_searching(data, self.grouping_role)
        target_fields = self._field_searching(
            data, self.target_roles, search_types=self.search_types
        )
        if len(target_fields) != 2:
            target_fields += self._field_searching(
                data,
                self.target_roles,
                search_types=self.search_types,
                space=SpaceEnum.additional,
            )
        return group_field, target_fields

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_fields: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict:
        if target_fields is None or len(target_fields) != 2:
            raise ValueError(
                "This operator works with 2 targets, but got {}".format(
                    len(target_fields) if target_fields else None
                )
            )
        result = {}
        for group, group_data in grouping_data:
            result[group[0]] = cls._inner_function(
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
