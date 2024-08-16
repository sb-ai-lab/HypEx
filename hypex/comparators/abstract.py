import warnings
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
    NAME_BORDER_SYMBOL,
)
from hypex.utils.adapter import Adapter
from hypex.utils.errors import (
    AbstractMethodError,
    FieldNotSuitableFieldError,
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

    def _get_fields(self, data: ExperimentData) -> Dict[str, Union[str, List[str]]]:
        tmp_role = True if data.ds.tmp_roles else False
        group_field = self._field_searching(
            data=data,
            roles=self.grouping_role,
        )
        target_fields = self._field_searching(
            data=data,
            roles=TempTargetRole() if tmp_role else self.target_roles,
            tmp_role=tmp_role,
            search_types=self.search_types,
        )
        baseline_field = self._field_searching(
            data=data,
            roles=self.baseline_role,
            tmp_role=tmp_role,
        )
        return {
            "group_field": group_field,
            "target_fields": target_fields,
            "baseline_field": baseline_field,
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
        grouping_data: Union[Tuple[List[Tuple[str, Dataset]]], Dict[str, Dataset]],
        compare_by: Literal["groups", "columns", "columns_in_groups", "cross"],
        target_fields: List[str],
        baseline_field: Optional[str],
    ):
        if isinstance(grouping_data, Tuple):
            baseline_data = Adapter.to_list(grouping_data[0])
            compared_data = Adapter.to_list(grouping_data[1])
        elif isinstance(grouping_data, Dict):
            compared_data = [(item[0], item[1]) for item in grouping_data.items()]
            baseline_data = [compared_data.pop()]
        else:
            raise TypeError(
                f"Grouping data must be tuple or list or tuple of lists, but got {type(grouping_data)}"
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

    @classmethod
    def _split_for_groups_mode(
        cls, data: Dataset, group_field: str, target_fields: List[str]
    ):
        if isinstance(target_fields, List) and len(target_fields) > 1:
            warnings.warn(
                f"Only one Target field can be passed when the comparison is done by groups. {len(target_fields)} passed. {target_fields[0]} will be used.",
            )
            target_fields = target_fields[0]
        if not isinstance(group_field, str):
            raise TypeError(
                f"group_field must be one string, {type(group_field)} passed."
            )

        data_buckets = sorted(data.groupby(by=group_field, fields_list=target_fields))
        baseline_data = cls._split_ds_into_columns([data_buckets.pop(0)])
        compared_data = cls._split_ds_into_columns(data=data_buckets)
        return baseline_data, compared_data

    @classmethod
    def _split_for_columns_mode(
        cls, data: Dataset, baseline_field: Optional[str], target_fields: List[str]
    ):
        if baseline_field is None:
            raise NoRequiredArgumentError("baseline_field")

        baseline_data = [(f"{baseline_field}", data[baseline_field])]
        compared_data = [(f"{column}", data[column]) for column in target_fields]
        return baseline_data, compared_data

    @classmethod
    def _split_for_columns_in_groups_mode(
        cls,
        data: Dataset,
        group_field: Optional[str],
        baseline_field: Optional[str],
        target_fields: List[str],
    ):
        if baseline_field is None:
            raise NoRequiredArgumentError("baseline_field")
        if group_field is None:
            raise NoRequiredArgumentError("group_field")
        if not isinstance(group_field, str):
            raise TypeError(
                f"group_field must be one string, {type(group_field)} passed."
            )
        if len(target_fields) > 1:
            warnings.warn(
                f"Too many fields passed as Target fields {target_fields}, only {target_fields[0]} will be used"
            )

        baseline_data = cls._split_ds_into_columns(
            data.groupby(by=group_field, fields_list=baseline_field)
        )
        compared_data = data.groupby(by=group_field, fields_list=target_fields[0])
        compared_data = cls._split_ds_into_columns(data=compared_data)
        return baseline_data, compared_data

    @classmethod
    def _split_for_cross_mode(
        cls,
        data: Dataset,
        group_field: Optional[str],
        baseline_field: Optional[str],
        target_fields: List[str],
    ):
        if baseline_field is None:
            raise NoRequiredArgumentError("baseline_field")
        if group_field is None:
            raise NoRequiredArgumentError("group_field")
        if not isinstance(group_field, str):
            raise TypeError(
                f"group_field must be one string, {type(group_field)} passed."
            )
        data_buckets = sorted(data.groupby(by=group_field, fields_list=baseline_field))
        baseline_data = cls._split_ds_into_columns([data_buckets.pop(0)])
        compared_data = sorted(data.groupby(by=group_field, fields_list=target_fields))
        compared_data.pop(0)
        compared_data = cls._split_ds_into_columns(data=compared_data)
        return baseline_data, compared_data

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

        target_fields = Adapter.to_list(target_fields)
        if compare_by == "groups":
            baseline_data, compared_data = cls._split_for_groups_mode(
                data, group_field, target_fields
            )
        elif compare_by == "columns":
            baseline_data, compared_data = cls._split_for_columns_mode(
                data, baseline_field, target_fields
            )
        elif compare_by == "columns_in_groups":
            baseline_data, compared_data = cls._split_for_columns_in_groups_mode(
                data, group_field, baseline_field, target_fields
            )
        elif compare_by == "cross":
            baseline_data, compared_data = cls._split_for_cross_mode(
                data, group_field, baseline_field, target_fields
            )
        else:
            raise ValueError(
                f"Wrong compare_by argument passed {compare_by}. It can be only one of the following modes: 'groups', 'columns', 'columns_in_groups', 'cross'."
            )
        return baseline_data, compared_data

    @classmethod
    def calc(
        cls,
        data: Dataset,
        compare_by: Literal["groups", "columns", "columns_in_groups", "cross"],
        target_fields: Union[str, List[str]],
        baseline_field: Optional[str] = None,
        group_field: Optional[str] = None,
        grouping_data: Optional[Tuple[List[Tuple[str, Dataset]]]] = None,
        **kwargs,
    ) -> Dict:
        target_fields = Adapter.to_list(target_fields)

        if grouping_data is None:
            grouping_data = cls._split_data_to_buckets(
                data=data,
                compare_by=compare_by,
                target_fields=target_fields,
                baseline_field=baseline_field,
                group_field=group_field,
            )

        if len(grouping_data[0]) < 1 or len(grouping_data[1]) < 1:
            raise FieldNotSuitableFieldError(group_field, "Grouping")

        baseline_data, compared_data = grouping_data
        return cls._execute_inner_function(
            baseline_data=baseline_data,
            compared_data=compared_data,
            **kwargs,
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        fields = self._get_fields(data)

        group_field = fields["group_field"]
        target_fields = fields["target_fields"]
        baseline_field = fields["baseline_field"]

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
            raise FieldNotSuitableFieldError(group_field, "Grouping")

        if (
            group_field[0] in data.groups
        ):  # TODO: propper split between groups and columns
            grouping_data = self._grouping_data_split(
                grouping_data=data.groups[group_field[0]],
                compare_by=self.compare_by,
                target_fields=target_fields,
                baseline_field=baseline_field,
            )
        else:
            grouping_data = None
            data.groups[group_field[0]] = {
                f"{group}": ds for group, ds in data.ds.groupby(group_field[0])
            }

        compare_result = self.calc(
            data=data.ds,
            compare_by=self.compare_by,
            target_fields=target_fields,
            baseline_field=Adapter.list_to_single(baseline_field),
            group_field=Adapter.list_to_single(group_field),
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
