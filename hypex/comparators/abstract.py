from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence, Literal

from hypex.dataset import (
    ABCRole,
    Dataset,
    ExperimentData,
    StatisticRole,
    TempTargetRole,
    InfoRole,
    DatasetAdapter,
    GroupingRole,
    TargetRole,
    PreTargetRole,
)
from hypex.executor import Calculator
from hypex.utils import (
    BackendsEnum,
    ExperimentDataEnum,
    FromDictTypes,
    SpaceEnum,
    NAME_BORDER_SYMBOL,
)
from hypex.utils.adapter import Adapter
from hypex.utils.errors import (
    AbstractMethodError,
    ComparisonNotSuitableFieldError,
    NoRequiredArgumentError,
    NoColumnsError,
)


class Comparator(Calculator):
    def __init__(
        self,
        compare_by: Literal["groups", "columns", "columns_in_groups", "cross"],
        grouping_role: Optional[ABCRole] = None,
        target_roles: Union[ABCRole, List[ABCRole], None] = None,
        baseline_role: Optional[ABCRole] = None,
        space: SpaceEnum = SpaceEnum.auto,
        key: Any = "",
    ):
        self.__additional_mode = space == SpaceEnum.additional
        super().__init__(key=key)
        self.grouping_role = grouping_role or GroupingRole()
        self.space = space
        self.compare_by = compare_by
        self.target_roles = target_roles or TargetRole()
        self.baseline_role = baseline_role or PreTargetRole()

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
        group_field = self._field_searching(
            data=data,
            field=self.grouping_role,
            search_types=self.search_types,
            space=SpaceEnum.data,
        )
        target_fields = self._field_searching(
            data,
            TempTargetRole(),
            # tmp_role=True,
            search_types=self.search_types,
            space=SpaceEnum.data,
        )
        baseline_field = self._field_searching(
            data=data, field=self.baseline_role, space=SpaceEnum.data
        )
        return group_field, target_fields, baseline_field

    @classmethod
    def _execute_inner_function(
        cls,
        baseline_data: List[Tuple[str, Dataset]],
        compared_data: List[Tuple[str, Dataset]],
        compare_by: Literal["groups", "columns", "columns_in_groups", "cross"],
        **kwargs,
    ) -> Dict:
        result = {}
        for baseline in baseline_data:
            #     result[baseline[0]] = {}
            for compared in compared_data:
                if (
                    compare_by != "columns_in_groups"
                    or baseline[0].split(NAME_BORDER_SYMBOL)[0]
                    == compared[0].split(NAME_BORDER_SYMBOL)[0]
                ):  # this checks if compared data are in the same group for columns_in_groups mode
                    result[compared[0]] = DatasetAdapter.to_dataset(
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
    def _split_ds_into_columns(
        data: List[Tuple[str, Dataset]]
    ) -> List[Tuple[str, Dataset]]:
        result = []
        for bucket in data:
            for column in bucket[1].columns:
                result.append(
                    (f"{bucket[0]}{NAME_BORDER_SYMBOL}{column}", bucket[1][column])
                )
        return result

    @classmethod
    def _split_data_to_buckets(
        cls,
        data: Dataset,
        compare_by: Literal["groups", "columns", "columns_in_groups", "cross"],
        target_fields: Union[str, List[str]],
        baseline_field: Optional[str] = None,
        group_field: Optional[str] = None,
    ) -> Tuple:
        """
        Splits the given dataset into buckets into baseline and compared data, based on the specified comparison mode.

        Args:
            data (Dataset): The dataset to be split.
            group_field (Union[Sequence[str], str]): The field(s) to group the data by.
            target_fields (Union[str, List[str]]): The field(s) to target for comparison.
            compare_by (Literal['groups', 'columns', 'columns_in_groups', 'cross'], optional): The method to compare the data. Defaults to 'groups'.
            baseline_field (Optional[str], optional): The column to use as the baseline for comparison. Required if `compare_by` is 'columns' or 'columns_in_groups'. Defaults to None.

        Returns:
            Tuple: A tuple containing the baseline data and the compared data.

        Raises:
            NoRequiredArgumentError: If `baseline_field` is None and `compare_by` is 'columns' or 'columns_in_groups' or 'cross'.
            ValueError: If `compare_by` is not one of the allowed values.
        """
        if compare_by == "groups":
            if isinstance(target_fields, List):
                target_fields = Adapter.list_to_single(target_fields)
            if not isinstance(target_fields, str):
                raise TypeError(
                    f"group_field must be one string, {type(group_field)} passed."
                )
        elif baseline_field is None:
            raise NoRequiredArgumentError("baseline_field")

        if compare_by != "columns":
            if group_field is None:
                raise NoRequiredArgumentError("group_field")
            elif isinstance(group_field, List):
                group_field = Adapter.list_to_single(group_field)
            if not isinstance(group_field, str):
                raise TypeError(
                    f"group_field must be one string, {type(group_field)} passed."
                )

        target_fields = Adapter.to_list(target_fields)
        if compare_by == "groups":
            data_buckets = data.groupby(by=group_field, fields_list=target_fields)
            baseline_data = cls._split_ds_into_columns([data_buckets.pop(0)])
            compared_data = cls._split_ds_into_columns(data=data_buckets)
        elif compare_by == "columns":
            baseline_data = [
                (f"0{NAME_BORDER_SYMBOL}{baseline_field}", data[baseline_field])
            ]
            compared_data = [
                (f"0{NAME_BORDER_SYMBOL}{column}", data[column])
                for column in target_fields
            ]
        elif compare_by == "columns_in_groups":
            baseline_data = cls._split_ds_into_columns(
                data.groupby(by=group_field, fields_list=baseline_field)
            )
            compared_data = data.groupby(by=group_field, fields_list=target_fields)
            compared_data = cls._split_ds_into_columns(data=compared_data)
        elif compare_by == "cross":
            data_buckets = data.groupby(by=group_field, fields_list=baseline_field)
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
        compare_by: Literal["groups", "columns", "columns_in_groups", "cross"],  # check if it is possible to make it mandatory
        target_fields: Union[str, List[str], None] = None,
        baseline_field: Optional[str] = None,
        group_field: Optional[str] = None,
        grouping_data: Optional[Tuple[List[Tuple[str, Dataset]]]] = None,
        **kwargs,
    ) -> Dict:
        target_fields = Adapter.to_list(target_fields)
        baseline_field = Adapter.list_to_single(baseline_field)
        group_field = Adapter.list_to_single(group_field)

        if compare_by == "columns_in_groups" and len(target_fields) > 1:
            raise ComparisonNotSuitableFieldError("target_fields")

        if grouping_data is None:
            grouping_data = cls._split_data_to_buckets(
                data=data,
                compare_by=compare_by,
                target_fields=target_fields,
                baseline_field=baseline_field,
                group_field=group_field,
            )
        if len(grouping_data[0]) < 1 or len(grouping_data[1]) < 1:
            raise ComparisonNotSuitableFieldError("group_field")
        baseline_data, compared_data = grouping_data
        return cls._execute_inner_function(
            baseline_data=baseline_data,
            compared_data=compared_data,
            compare_by=compare_by,
            **kwargs,
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields, baseline_field = self._get_fields(data)

        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )
        if not target_fields:
            if (
                data.ds.tmp_roles
            ):  # если колонка не подходит для теста, то тагет будет пустой, но если есть темп роли, то это нормальное поведение
                return data
            else:
                raise NoColumnsError(TargetRole().role_name)
        if not group_field and self.compare_by != "columns":
            raise ComparisonNotSuitableFieldError("group_field")

        # if group_field[0] in data.groups:  # TODO: to recheck if this is a correct check
        #     grouping_data = list(data.groups[group_field[0]].items())
        # else:
        #     grouping_data = None
        #     data.groups[group_field[0]] = {
        #         f"{int(group)}": ds for group, ds in data.ds.groupby(group_field[0])
        #     }
        compare_result = self.calc(
            data=data.ds,
            compare_by=self.compare_by,
            target_fields=target_fields,
            baseline_field=baseline_field,
            group_field=group_field,
        )
        result_dataset = self._local_extract_dataset(
            compare_result, {key: StatisticRole() for key in compare_result}
        )
        return self._set_value(data, result_dataset)


class StatHypothesisTesting(Comparator, ABC):
    def __init__(
        self,
        compare_by: Literal["groups", "columns", "columns_in_groups", "cross"],
        grouping_role: Union[ABCRole, None] = None,
        space: SpaceEnum = SpaceEnum.auto,
        reliability: float = 0.05,
        key: Any = "",
    ):
        super().__init__(
            compare_by=compare_by, grouping_role=grouping_role, space=space, key=key
        )
        self.reliability = reliability
