from typing import Optional, List, Any

from hypex.dataset import ABCRole, TargetRole, Dataset, ExperimentData
from hypex.dataset.roles import MatchingRole
from hypex.executor import Calculator
from hypex.utils import ExperimentDataEnum


class SMD(Calculator):
    def __init__(
        self,
        searched_roles: Optional[List[ABCRole]] = None,
        key: Any = "",
        random_state: Optional[int] = None,
    ):
        super().__init__(key, random_state)
        self.searched_roles = searched_roles or [TargetRole(), MatchingRole()]

    @classmethod
    def calc(
        cls,
        data: Dataset,
        other: Optional[Dataset] = None,
        searched_roles: Optional[List[ABCRole]] = None,
        **kwargs
    ):
        return cls._inner_function(
            data, other=other, searched_roles=searched_roles, **kwargs
        )

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        other: Optional[Dataset] = None,
        searched_roles: Optional[List[ABCRole]] = None,
        **kwargs
    ) -> Any:
        if len(searched_roles) != 2:
            raise ValueError  # TODO
        original_data = data[data.search_columns(searched_roles[0])]
        matched_data = data[data.search_columns(searched_roles[1])]
        resul = (original_data.mean() + matched_data.mean()) / original_data.std()
        return resul

    def _set_value(
        self, data: ExperimentData, value: Any, key: Any = None
    ) -> ExperimentData:
        return data.set_value(ExperimentDataEnum.variables, str(self.id), "SMD", value)

    def execute(self, data: ExperimentData) -> ExperimentData:
        other_data_index = data.additional_fields.search_columns(self.searched_roles)
        other_data = None
        if len(other_data_index) > 0:
            other_data = data.ds.loc[other_data_index]
        return self._set_value(
            data,
            self.calc(data.ds, other=other_data, searched_roles=self.searched_roles),
            key=self.id,
        )
