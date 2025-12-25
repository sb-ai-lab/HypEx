from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Literal

from ..dataset import (
    ABCRole,
    AdditionalTargetRole,
    Dataset,
    DatasetAdapter,
    ExperimentData,
    GroupingRole,
    InfoRole,
    PreTargetRole,
    StatisticRole,
    TargetRole,
    TempTargetRole,
)
from ..executor import Calculator
from ..utils import (
    NAME_BORDER_SYMBOL,
    BackendsEnum,
    ExperimentDataEnum,
    FromDictTypes,
    GroupingDataType,
)
from ..utils.errors import (
    AbstractMethodError,
    NoColumnsError,
    NoRequiredArgumentError,
    NotSuitableFieldError,
)


class Comparator(Calculator, ABC):
    def __init__(
        self,
        compare_by: Literal[
            "groups", "columns", "columns_in_groups", "cross", "matched_pairs"
        ],
        grouping_role: ABCRole | None = None,
        target_roles: ABCRole | list[ABCRole] | None = None,
        baseline_role: ABCRole | None = None,
        key: Any = "",
        calc_kwargs: dict[str, Any] = {},
    ):
        super().__init__(key=key)
        self.grouping_role = grouping_role or GroupingRole()
        self.compare_by = compare_by
        self.target_roles = target_roles or TargetRole()
        self.baseline_role = baseline_role or PreTargetRole()
        self.calc_kwargs = calc_kwargs

    @property
    def search_types(self) -> list[type] | None:
        return None

    def _local_extract_dataset(
        self, compare_result: dict[Any, Any], roles: dict[Any, ABCRole]
    ) -> Dataset:
        return self._extract_dataset(compare_result, roles)

    @classmethod
    @abstractmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> Any:
        raise AbstractMethodError

    def _get_fields_data(self, data: ExperimentData) -> dict[str, Dataset]:
        tmp_role = (
            True if data.ds.tmp_roles or data.additional_fields.tmp_roles else False
        )
        group_field_data = data.field_data_search(roles=self.grouping_role)
        target_fields_data = data.field_data_search(
            roles=(
                (TempTargetRole() if data.ds.tmp_roles else AdditionalTargetRole())
                if tmp_role
                else self.target_roles
            ),
            tmp_role=tmp_role,
            search_types=self.search_types,
        )
        baseline_field_data = data.field_data_search(roles=self.baseline_role)
        return {
            "group_field": group_field_data,
            "target_fields": target_fields_data,
            "baseline_field": baseline_field_data,
        }

    @classmethod
    def _execute_inner_function(
        cls,
        baseline_data: list[tuple[str, Dataset]],
        compared_data: list[tuple[str, Dataset]],
        compare_by: Literal[
            "groups", "columns", "columns_in_groups", "cross", "matched_pairs"
        ],
        **kwargs,
    ) -> dict:
        result = {}
        for i in range(len(compared_data)):
            res_name = (
                compared_data[i][0]
                if compare_by == "groups"
                else f"{compared_data[i][0]}{NAME_BORDER_SYMBOL}{compared_data[i][1].columns[0]}"
            )
            result[res_name] = DatasetAdapter.to_dataset(
                cls._inner_function(
                    baseline_data[0 if len(baseline_data) == 1 else i][1],
                    compared_data[i][1],
                    **kwargs,
                ),
                InfoRole(),
            )
        return result

    @staticmethod
    def _check_test_data(test_data: Dataset | None = None) -> Dataset:
        if test_data is None:
            raise ValueError("test_data is needed for evaluation")
        return test_data

    def _set_value(
        self, data: ExperimentData, value: Dataset | None = None, key: Any = None
    ) -> ExperimentData:
        data.set_value(
            ExperimentDataEnum.analysis_tables,
            self.id,
            value,
        )
        return data

    @staticmethod
    def _extract_dataset(
        compare_result: FromDictTypes, roles: dict[Any, ABCRole]
    ) -> Dataset:
        if isinstance(next(iter(compare_result.values())), Dataset):
            cr_list_v: list[Dataset] = list(compare_result.values())
            result = cr_list_v[0]
            if len(cr_list_v) > 1:
                result = result.append(cr_list_v[1:])
            result.index = list(compare_result.keys())
            return result
        return Dataset.from_dict(compare_result, roles, BackendsEnum.pandas)

    @staticmethod
    def _grouping_data_split(
        grouping_data: dict[str, Dataset],
        compare_by: Literal[
            "groups", "columns", "columns_in_groups", "cross", "matched_pairs"
        ],
        target_fields: list[str],
        baseline_field: str | None = None,
    ) -> GroupingDataType:
        if not isinstance(grouping_data, dict):
            raise TypeError(
                f"Grouping data must be dict of strings and datasets, but got {type(grouping_data)}"
            )

        compared_data = list(grouping_data.items())
        baseline_data = [compared_data.pop(0)]
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
        data: list[tuple[str, Dataset]],
    ) -> list[tuple[str, Dataset]]:
        result = [
            (bucket[0], bucket[1][column])
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
        compare_by: Literal[
            "groups", "columns", "columns_in_groups", "cross", "matched_pairs"
        ],
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
    def _split_for_matched_pairs_mode(
        cls,
        group_field_data: Dataset,
        baseline_field_data: Dataset,
        target_fields_data: Dataset,
    ) -> GroupingDataType:
        group_field_data = cls._field_validity_check(
            group_field_data, "group_field_data", "matched_pairs"
        )
        baseline_field_data = cls._field_validity_check(
            baseline_field_data, "baseline_field_data", "matched_pairs"
        )
        target_fields_data = cls._field_validity_check(
            target_fields_data, "target_fields_data", "matched_pairs"
        )

        compared_data = target_fields_data.groupby(by=group_field_data)
        baseline_indexes = baseline_field_data.groupby(by=group_field_data)
        baseline_data = []

        # mapping the data of the baseline data to its matches data. If there are no matches, matching index will be -1
        for group in baseline_indexes:
            name = group[0]
            indexes = group[1].iget_values(column=0)
            dummy_index = target_fields_data.index[-1]
            indexes = list(map(lambda x: dummy_index if x < 0 else x, indexes))
            baseline_data.append((name, target_fields_data.loc[indexes, :]))

        return baseline_data, compared_data

    @classmethod
    def _split_data_to_buckets(
        cls,
        compare_by: Literal[
            "groups", "columns", "columns_in_groups", "cross", "matched_pairs"
        ],
        target_fields_data: Dataset,
        baseline_field_data: Dataset,
        group_field_data: Dataset,
    ) -> GroupingDataType:
        """
        Splits the given dataset into buckets into baseline and compared data, based on the specified comparison mode.

        Args:
            group_field (Union[Sequence[str], str]): The field(s) to group the data by.
            target_fields (Union[str, List[str]]): The field(s) to target for comparison.
            compare_by (Literal['groups', 'columns', 'columns_in_groups', 'cross', 'matched_pairs'], optional): The method to compare the data. Defaults to 'groups'.
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
        elif compare_by == "matched_pairs":
            baseline_data, compared_data = cls._split_for_matched_pairs_mode(
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
        compare_by: (
            Literal["groups", "columns", "columns_in_groups", "cross", "matched_pairs"]
            | None
        ) = None,
        target_fields_data: Dataset | None = None,
        baseline_field_data: Dataset | None = None,
        group_field_data: Dataset | None = None,
        grouping_data: (
            tuple[list[tuple[str, Dataset]]] | list[tuple[str, Dataset]] | None
        ) = None,
        **kwargs,
    ) -> dict:
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
            compare_by=compare_by,
            **kwargs,
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        """
        Execute the comparator on the given data.

        The comparator will split the data into a baseline and a comparison
        dataset based on the compare_by argument. Then it will calculate
        statistics comparing the baseline and comparison datasets.

        :param data: The ExperimentData to execute the comparator on
        :type data: ExperimentData
        :return: The ExperimentData with the comparison results
        :rtype: ExperimentData
        """
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
            # If the column is not suitable for the test, then the target will be empty, but if there is a role tempo, then this is normal behavior
            if data.ds.tmp_roles:
                return data
            else:
                raise NoColumnsError(TargetRole().role_name)

        if len(group_field_data.columns) != 1 and self.compare_by != "columns":
            raise NotSuitableFieldError(group_field_data, "Grouping")

        if (
            group_field_data.columns[0] in data.groups
        ) and self.compare_by != "matched_pairs":  # TODO: proper split between groups and columns
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
            combined_data = (
                data.ds.merge(
                    data.additional_fields[
                        [
                            col
                            for col in data.additional_fields.columns
                            if isinstance(
                                data.additional_fields.roles[col], AdditionalTargetRole
                            )
                        ]
                    ],
                    left_index=True,
                    right_index=True,
                    how="outer",
                )
                if any(
                    isinstance(data.additional_fields.roles[col], AdditionalTargetRole)
                    for col in data.additional_fields.columns
                )
                else data.ds
            )

            data.groups[group_field_data.columns[0]] = {
                f"{group}": ds for group, ds in combined_data.groupby(group_field_data)
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
            **self.calc_kwargs,
            compare_by=self.compare_by,
            target_fields_data=target_fields_data,
            baseline_field_data=baseline_field_data,
            group_field_data=group_field_data,
            grouping_data=grouping_data,
            # kwargs=,
        )
        result_dataset = self._local_extract_dataset(
            compare_result, {key: StatisticRole() for key in compare_result}
        )
        return self._set_value(data, result_dataset)


class StatHypothesisTesting(Comparator, ABC):
    def __init__(
        self,
        compare_by: Literal[
            "groups", "columns", "columns_in_groups", "cross", "matched_pairs"
        ],
        grouping_role: ABCRole | None = None,
        target_role: ABCRole | None = None,
        baseline_role: ABCRole | None = None,
        reliability: float = 0.05,
        key: Any = "",
        calc_kwargs: dict[str, Any] = {},
    ):
        super().__init__(
            compare_by=compare_by,
            grouping_role=grouping_role,
            target_roles=target_role,
            baseline_role=baseline_role,
            key=key,
            calc_kwargs=calc_kwargs,
        )
        self.reliability = reliability
