import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence, Literal

from ..dataset import (
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
from ..executor import Calculator
from ..utils import (
    BackendsEnum,
    ExperimentDataEnum,
    FromDictTypes,
    NAME_BORDER_SYMBOL,
    GroupingDataType,
)
from ..utils.adapter import Adapter
from ..utils.errors import (
    AbstractMethodError,
    NotSuitableFieldError,
    NoRequiredArgumentError,
    NoColumnsError,
)


class Comparator(Calculator, ABC):
    def __init__(
        self,
        compare_by: Literal["groups", "columns", "columns_in_groups", "cross"],
        grouping_role: Optional[ABCRole] = None,
        target_roles: Union[ABCRole, List[ABCRole], None] = None,
        baseline_role: Optional[ABCRole] = None,
        key: Any = "",
    ):
        super().__init__(key=key)
        self.grouping_role = grouping_role or GroupingRole()
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

    def _get_fields_data(self, data: ExperimentData) -> Dict[str, Dataset]:
        tmp_role = True if data.ds.tmp_roles else False
        group_field_data = data.field_data_search(roles=self.grouping_role)
        target_fields_data = data.field_data_search(
            roles=TempTargetRole() if tmp_role else self.target_roles,
            tmp_role=tmp_role,
            search_types=self.search_types,
        )
        baseline_field_data = data.field_data_search(
            roles=self.baseline_role, tmp_role=tmp_role
        )
        return {
            "group_field": group_field_data,
            "target_fields": target_fields_data,
            "baseline_field": baseline_field_data,
        }

    @classmethod
    def _execute_inner_function(
        cls,
        baseline_data: List[Tuple[str, Dataset]],
        compared_data: List[Tuple[str, Dataset]],
        **kwargs,
    ) -> Dict:
        result = {}
        for i in range(len(compared_data)):
            result[compared_data[i][0]] = DatasetAdapter.to_dataset(
                cls._inner_function(
                    baseline_data[0 if len(baseline_data) == 1 else i][1],
                    compared_data[i][1],
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
            value,
        )
        return data

    @staticmethod
    def _extract_dataset(
        compare_result: FromDictTypes, roles: Dict[Any, ABCRole]
    ) -> Dataset:
        if isinstance(list(compare_result.values())[0], Dataset):
            cr_list_v: List[Dataset] = list(compare_result.values())
            result = cr_list_v[0]
            if len(cr_list_v) > 1:
                result = result.append(cr_list_v[1:])
            result.index = list(compare_result.keys())
            return result
        return Dataset.from_dict(compare_result, roles, BackendsEnum.pandas)

    # TODO выделить в отдельную функцию с кваргами (нужно для альфы)

    @staticmethod
    def _grouping_data_split(
        grouping_data: Dict[str, Dataset],
        compare_by: Literal["groups", "columns", "columns_in_groups", "cross"],
        target_fields: List[str],
        baseline_field: Optional[str],
    ) -> GroupingDataType:
        if isinstance(grouping_data, Dict):
            compared_data = [(name, data) for name, data in grouping_data.items()]
            baseline_data = [compared_data.pop(0)]
        else:
            raise TypeError(
                f"Grouping data must be dict of strings and datasets, but got {type(grouping_data)}"
            )

        baseline_data = [
            (
                bucket[0],
                bucket[1][target_fields if compare_by == "groups" else baseline_field],
            )
            for bucket in baseline_data
        ]
        compared_data = [
            (bucket[0], bucket[1][target_fields]) for bucket in compared_data
        ]

        return baseline_data, compared_data

    @staticmethod
    def _split_ds_into_columns(
        data: List[Tuple[str, Dataset]]
    ) -> List[Tuple[str, Dataset]]:
        result = [
            (f"{bucket[0]}{NAME_BORDER_SYMBOL}{column}", bucket[1][column])
            for bucket in data
            for column in bucket[1].columns
        ]

        return result

    @staticmethod
    def _field_validity_check(
        field_data: Dataset,
        comparison_role: Literal[
            "group_field_data", "target_fields_data", "baseline_field_data"
        ],
        compare_by: Literal["groups", "columns", "columns_in_groups", "cross"],
    ) -> Dataset:
        if len(field_data.columns) == 0:
            raise NoRequiredArgumentError(comparison_role)
        if len(field_data.columns) > 1:
            warnings.warn(
                f"{comparison_role} must have only one column when the comparison is done by {compare_by}. {len(field_data.columns)} passed. {field_data.columns[0]} will be used.",
            )
            field_data = field_data[field_data.columns[0]]

        return field_data

    @classmethod
    def _split_for_groups_mode(
        cls,
        group_field_data: Dataset,
        target_fields_data: Dataset,
    ) -> GroupingDataType:
        target_fields_data = cls._field_validity_check(
            target_fields_data, "target_fields_data", "groups"
        )
        group_field_data = cls._field_validity_check(
            group_field_data, "group_field_data", "groups"
        )

        data_buckets = sorted(
            target_fields_data.groupby(by=group_field_data), key=lambda tup: tup[0]
        )

        baseline_data = cls._split_ds_into_columns([data_buckets.pop(0)])
        compared_data = cls._split_ds_into_columns(data=data_buckets)

        return baseline_data, compared_data

    @classmethod
    def _split_for_columns_mode(
        cls,
        baseline_field_data: Dataset,
        target_fields_data: Dataset,
    ) -> GroupingDataType:
        baseline_field_data = cls._field_validity_check(
            baseline_field_data, "baseline_field_data", "columns"
        )
        if len(target_fields_data.columns) == 0:
            raise NoRequiredArgumentError(target_fields_data)

        baseline_data = [(f"{baseline_field_data.columns[0]}", baseline_field_data)]

        compared_data = [
            (f"{column}", target_fields_data[column])
            for column in target_fields_data.columns
        ]

        return baseline_data, compared_data

    @classmethod
    def _split_for_columns_in_groups_mode(
        cls,
        group_field_data: Dataset,
        baseline_field_data: Dataset,
        target_fields_data: Dataset,
    ) -> GroupingDataType:
        baseline_field_data = cls._field_validity_check(
            baseline_field_data, "baseline_field_data", "columns_in_groups"
        )
        target_fields_data = cls._field_validity_check(
            target_fields_data, "target_fields_data", "columns_in_groups"
        )
        group_field_data = cls._field_validity_check(
            group_field_data, "group_field_data", "columns_in_groups"
        )

        baseline_data = baseline_field_data.groupby(by=group_field_data)

        compared_data = cls._split_ds_into_columns(
            target_fields_data.groupby(by=group_field_data)
        )

        return baseline_data, compared_data

    @classmethod
    def _split_for_cross_mode(
        cls,
        group_field_data: Dataset,
        baseline_field_data: Dataset,
        target_fields_data: Dataset,
    ) -> GroupingDataType:
        baseline_field_data = cls._field_validity_check(
            baseline_field_data, "baseline_field_data", "cross"
        )
        target_fields_data = cls._field_validity_check(
            target_fields_data, "target_fields_data", "cross"
        )
        group_field_data = cls._field_validity_check(
            group_field_data, "group_field_data", "cross"
        )

        baseline_data = [
            sorted(
                baseline_field_data.groupby(by=group_field_data), key=lambda tup: tup[0]
            ).pop(0)
        ]

        compared_data = sorted(
            target_fields_data.groupby(by=group_field_data), key=lambda tup: tup[0]
        )
        compared_data.pop(0)
        compared_data = cls._split_ds_into_columns(data=compared_data)

        return baseline_data, compared_data

    @classmethod
    def _split_data_to_buckets(
        cls,
        compare_by: Literal["groups", "columns", "columns_in_groups", "cross"],
        target_fields_data: Dataset,
        baseline_field_data: Dataset,
        group_field_data: Dataset,
    ) -> GroupingDataType:
        """
        Splits the given dataset into buckets into baseline and compared data, based on the specified comparison mode.

        Args:
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
            baseline_data, compared_data = cls._split_for_groups_mode(
                group_field_data, target_fields_data
            )
        elif compare_by == "columns":
            baseline_data, compared_data = cls._split_for_columns_mode(
                baseline_field_data, target_fields_data
            )
        elif compare_by == "columns_in_groups":
            baseline_data, compared_data = cls._split_for_columns_in_groups_mode(
                group_field_data, baseline_field_data, target_fields_data
            )
        elif compare_by == "cross":
            baseline_data, compared_data = cls._split_for_cross_mode(
                group_field_data, baseline_field_data, target_fields_data
            )
        else:
            raise ValueError(
                f"Wrong compare_by argument passed {compare_by}. It can be only one of the following modes: 'groups', 'columns', 'columns_in_groups', 'cross'."
            )
        return baseline_data, compared_data

    @classmethod
    def calc(
        cls,
        compare_by: Optional[
            Literal["groups", "columns", "columns_in_groups", "cross"]
        ] = None,
        target_fields_data: Optional[Dataset] = None,
        baseline_field_data: Optional[Dataset] = None,
        group_field_data: Optional[Dataset] = None,
        grouping_data: Optional[
            Tuple[List[Tuple[str, Dataset]], List[Tuple[str, Dataset]]]
        ] = None,
        **kwargs,
    ) -> Dict:

        if compare_by is None and target_fields_data is None:
            raise ValueError(
                "You should pass either compare_by or target_fields argument."
            )

        if grouping_data is None:
            grouping_data = cls._split_data_to_buckets(
                compare_by=compare_by,
                target_fields_data=target_fields_data,
                baseline_field_data=baseline_field_data,
                group_field_data=group_field_data,
            )

        baseline_data, compared_data = grouping_data
        return cls._execute_inner_function(
            baseline_data=baseline_data,
            compared_data=compared_data,
            **kwargs,
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        fields = self._get_fields_data(data)

        group_field_data = fields["group_field"]
        target_fields_data = fields["target_fields"]
        baseline_field_data = fields["baseline_field"]

        self.key = str(
            target_fields_data.columns[0]
            if len(target_fields_data.columns) == 1
            else (list(target_fields_data.columns) or "")
        )

        if len(target_fields_data.columns) == 0:
            if (
                data.ds.tmp_roles
            ):  # если колонка не подходит для теста, то тагет будет пустой, но если есть темп роли, то это нормальное поведение
                return data
            else:
                raise NoColumnsError(TargetRole().role_name)

        if len(group_field_data.columns) != 1 and self.compare_by != "columns":
            raise NotSuitableFieldError(group_field_data, "Grouping")

        if (
            group_field_data.columns[0] in data.groups
        ):  # TODO: proper split between groups and columns
            grouping_data = self._grouping_data_split(
                grouping_data=data.groups[group_field_data.columns[0]],
                compare_by=self.compare_by,
                target_fields=(
                    [data.ds.columns[0]]
                    if group_field_data.columns[0] == target_fields_data.columns[0]
                    else list(target_fields_data.columns)
                ),
                baseline_field=(
                    baseline_field_data.columns[0]
                    if len(baseline_field_data.columns) > 0
                    else None
                ),
            )
        else:
            data.groups[group_field_data.columns[0]] = {
                f"{group}": ds for group, ds in data.ds.groupby(group_field_data)
            }
            grouping_data = self._split_data_to_buckets(
                compare_by=self.compare_by,
                target_fields_data=target_fields_data,
                baseline_field_data=baseline_field_data,
                group_field_data=group_field_data,
            )

        if len(grouping_data[0]) < 1 or len(grouping_data[1]) < 1:
            raise NotSuitableFieldError(group_field_data, "Grouping")

        compare_result = self.calc(
            compare_by=self.compare_by,
            target_fields_data=target_fields_data,
            baseline_field_data=baseline_field_data,
            group_field_data=group_field_data,
            grouping_data=grouping_data,
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
        target_role: Union[ABCRole, None] = None,
        baseline_role: Union[ABCRole, None] = None,
        reliability: float = 0.05,
        key: Any = "",
    ):
        super().__init__(
            compare_by=compare_by,
            grouping_role=grouping_role,
            target_roles=target_role,
            baseline_role=baseline_role,
            key=key,
        )
        self.reliability = reliability
