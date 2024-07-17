from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from hypex.dataset import (
    ABCRole,
    Dataset,
    ExperimentData,
    StatisticRole,
    TempTargetRole,
    InfoRole,
    DatasetAdapter,
)
from hypex.executor import GroupCalculator
from hypex.utils import (
    BackendsEnum,
    ExperimentDataEnum,
    FromDictTypes,
    SpaceEnum,
)
from hypex.utils.errors import AbstractMethodError


class GroupComparator(GroupCalculator):
    def __init__(
        self,
        grouping_role: Optional[ABCRole] = None,
        space: SpaceEnum = SpaceEnum.auto,
        key: Any = "",
    ):
        self.__additional_mode = space == SpaceEnum.additional
        super().__init__(grouping_role=grouping_role, space=space, key=key)

    @property
    def search_types(self) -> Optional[List[type]]:
        return None

    def _local_extract_dataset(
        self, compare_result: Dict[Any, Any], roles: Dict[Any, ABCRole]
    ) -> Dataset:
        return self._extract_dataset(compare_result, roles)

    @classmethod
    @abstractmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Any:
        raise AbstractMethodError

    def _get_fields(self, data: ExperimentData):
        group_field = self._field_searching(data, self.grouping_role)
        target_fields = self._field_searching(
            data,
            TempTargetRole(),
            tmp_role=True,
            search_types=self.search_types,
            space=SpaceEnum.data,
        )
        return group_field, target_fields

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_fields: Optional[List[str]] = None,
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
            # TODO roles
            if target_fields:
                grouping_data[i][1].tmp_roles = old_data.tmp_roles
                result[result_key] = DatasetAdapter.to_dataset(
                    cls._inner_function(
                        data=grouping_data[0][1][target_fields],
                        test_data=grouping_data[i][1][target_fields],
                        **kwargs,
                    ),
                    InfoRole(),
                )
            else:
                result[result_key] = DatasetAdapter.to_dataset(
                    cls._inner_function(
                        grouping_data[0][1],
                        grouping_data[i][1],
                        **kwargs,
                    ),
                    InfoRole(),
                )
        return result

    @staticmethod
    def _check_test_data(test_data: Optional[Dataset] = None) -> Dataset:
        if test_data is None:
            raise ValueError("test_data is needed for evaluation")
        return test_data

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

    # TODO compare_result.values(), но тайпинг для нее FromDictTypes
    @staticmethod
    def _extract_dataset(
        compare_result: FromDictTypes, roles: Dict[Any, ABCRole]
    ) -> Dataset:
        if isinstance(list(compare_result.values())[0], Dataset):
            cr_list_v: List[Dataset] = list(compare_result.values())
            result = cr_list_v[0]
            if len(cr_list_v) > 1:
                result = result.append(cr_list_v[1:])
            return result
        return Dataset.from_dict(compare_result, roles, BackendsEnum.pandas)

    # TODO выделить в отдельную функцию с кваргами (нужно для альфы)
    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields = self._get_fields(data)
        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )
        if (
            not target_fields and data.ds.tmp_roles
        ):  # если колонка не подходит для теста, то тагет будет пустой, но если есть темп роли, то это нормальное поведение
            return data
        if group_field[0] in data.groups:  # TODO: to recheck if this is a correct check
            grouping_data = list(data.groups[group_field[0]].items())
        else:
            grouping_data = None
            data.groups[group_field[0]] = {
                f"{int(group)}": ds for group, ds in data.ds.groupby(group_field[0])
            }
        compare_result = self.calc(
            data=data.ds,
            group_field=group_field,
            grouping_data=grouping_data,
            target_fields=target_fields,
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
        reliability: float = 0.05,
        key: Any = "",
    ):
        super().__init__(grouping_role, space, key)
        self.reliability = reliability
