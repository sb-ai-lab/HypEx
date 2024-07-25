from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence, Literal

from hypex.dataset import (
    ABCRole,
    Dataset,
    ExperimentData,
    StatisticRole,
    TempTargetRole,
    InfoRole,
    DatasetAdapter, GroupingRole,
)
from hypex.executor import Calculator
from hypex.utils import (
    BackendsEnum,
    ExperimentDataEnum,
    FromDictTypes,
    SpaceEnum, NAME_BORDER_SYMBOL,
)
from hypex.utils.adapter import Adapter
from hypex.utils.errors import AbstractMethodError, ComparisonNotSuitableFieldError, NoRequiredArgumentError


class Comparator(Calculator):
    def __init__(
        self,
        grouping_role: Optional[ABCRole] = None,
        space: SpaceEnum = SpaceEnum.auto,
        key: Any = "",
    ):
        self.__additional_mode = space == SpaceEnum.additional
        super().__init__(key=key)
        self.grouping_role = grouping_role or GroupingRole()
        self.space = space

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
        baseline_data: Union[Tuple[str, Dataset], List[Tuple[str, Dataset]]],
        compared_data: Union[Tuple[str, Dataset], List[Tuple[str, Dataset]]],
        **kwargs,
    ) -> Dict:
        baseline_data = Adapter.to_list(baseline_data)
        compared_data = Adapter.to_list(compared_data)
        result = {}
        for baseline in baseline_data:
            result[baseline[0]] = {}
            for compared in compared_data:
                result[baseline[0]][compared[0]] = DatasetAdapter.to_dataset(
                    cls._inner_function(baseline[1], compared[1], **kwargs),
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

    @staticmethod
    def _split_ds_into_columns(data: List[Tuple[str, Dataset]]) -> List[Tuple[str, Dataset]]:
        result = []
        for bucket in data:
            for column in bucket[1].columns:
                result.append((f"{bucket[0]}{NAME_BORDER_SYMBOL}{column}", bucket[1][column]))
        return result

    @classmethod
    def split_data_to_buckets(
            cls,
            data: Dataset,
            target_fields: Union[str, List[str]],
            compare_by: Literal[
                "groups", "columns", "columns_in_groups", "cross"
            ] = "groups",
            group_field: Union[str, List[str], None] = None,
            baseline_column: Optional[str] = None,
    ) -> Tuple:
        """
        Splits the given dataset into buckets into baseline and compared data, based on the specified comparison mode.

        Args:
            data (Dataset): The dataset to be split.
            group_field (Union[Sequence[str], str]): The field(s) to group the data by.
            target_fields (Union[str, List[str]]): The field(s) to target for comparison.
            compare_by (Literal['groups', 'columns', 'columns_in_groups', 'cross'], optional): The method to compare the data. Defaults to 'groups'.
            baseline_column (Optional[str], optional): The column to use as the baseline for comparison. Required if `compare_by` is 'columns' or 'columns_in_groups'. Defaults to None.

        Returns:
            Tuple: A tuple containing the baseline data and the compared data.

        Raises:
            NoRequiredArgumentError: If `baseline_column` is None and `compare_by` is 'columns' or 'columns_in_groups' or 'cross'.
            ValueError: If `compare_by` is not one of the allowed values.
        """
        if compare_by == "groups":
            if isinstance(target_fields, List):
                target_fields = Adapter.list_to_single(target_fields)
            if not isinstance(target_fields, str):
                raise TypeError(f"group_field must be one string, {type(group_field)} passed.")
        elif baseline_column is None:
                raise NoRequiredArgumentError("baseline_column")

        if compare_by != "columns":
            if group_field is None:
                raise NoRequiredArgumentError("group_field")
            elif isinstance(group_field, List):
                group_field = Adapter.list_to_single(group_field)
            if not isinstance(group_field, str):
                raise TypeError(f"group_field must be one string, {type(group_field)} passed.")

        target_fields = Adapter.to_list(target_fields)
        if compare_by == "groups":
            data_buckets = data.groupby(by=group_field, fields_list=target_fields)
            baseline_data = cls._split_ds_into_columns([data_buckets.pop(0)])
            compared_data = cls._split_ds_into_columns(data=data_buckets)
        elif compare_by == "columns":
            baseline_data = [(f"0{NAME_BORDER_SYMBOL}{baseline_column}", data[baseline_column])]
            compared_data = [
                (f"0{NAME_BORDER_SYMBOL}{column}", data[column])
                for column in target_fields
                if column != baseline_column
            ]
        elif compare_by == "columns_in_groups":
            baseline_data = cls._split_ds_into_columns(data.groupby(
                by=group_field, fields_list=baseline_column
            ))
            compared_data = data.groupby(by=group_field, fields_list=target_fields)
            compared_data = cls._split_ds_into_columns(data=compared_data)
        elif compare_by == "cross":
            data_buckets = data.groupby(by=group_field, fields_list=baseline_column)
            baseline_data = cls._split_ds_into_columns([data_buckets.pop(0)])
            compared_data = data.groupby(by=group_field, fields_list=target_fields)
            compared_data.pop(0)
            compared_data = cls._split_ds_into_columns(data=compared_data)
        else:
            raise ValueError("compare_by")
        return baseline_data, compared_data

    @classmethod
    def calc(
            cls,
            data: Dataset,
            group_field: Union[Sequence[str], str, None] = None,
            grouping_data: Optional[List[Tuple[str, Dataset]]] = None,
            target_fields: Union[
                str, List[str], None
            ] = None,  # why is it possible to leave it None?
            baseline_column: Optional[str] = None,
            **kwargs,
    ) -> Dict:
        group_field = Adapter.to_list(group_field)

        if grouping_data is None:
            grouping_data = cls.split_data_to_buckets(
                data=data,
                group_field=group_field,
                target_fields=target_fields,
                baseline_column=baseline_column,
            )
        if len(grouping_data) < 2:
            raise ComparisonNotSuitableFieldError(group_field)
        baseline_data, compared_data = grouping_data
        return cls._execute_inner_function(
            baseline_data=baseline_data, compared_data=compared_data, **kwargs
        )

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


class StatHypothesisTesting(Comparator, ABC):
    def __init__(
        self,
        grouping_role: Union[ABCRole, None] = None,
        space: SpaceEnum = SpaceEnum.auto,
        reliability: float = 0.05,
        key: Any = "",
    ):
        super().__init__(grouping_role, space, key)
        self.reliability = reliability
