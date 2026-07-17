from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Literal

import numpy as np

from ..dataset import (
    ABCRole,
    AdditionalTargetRole,
    AdditionalTreatmentRole,
    Dataset,
    DatasetAdapter,
    ExperimentData,
    GroupingRole,
    SmallDataset,
    InfoRole,
    PreTargetRole,
    StatisticRole,
    TargetRole,
    TempTargetRole,
)
from ..dataset.abstract import GroupedDataset
from ..executor import Calculator
from ..utils import (
    NAME_BORDER_SYMBOL,
    BackendsEnum,
    ExperimentDataEnum,
    FromDictTypes,
    GroupingDataType,
    timeit
)
from ..utils.errors import (
    AbstractMethodError,
    NoColumnsError,
    NoRequiredArgumentError,
    NotSuitableFieldError,
)


class BaseComparator(Calculator, ABC):
    """
    Base class for all comparators. Owns role management, field resolution,
    and result storage. Does not prescribe how the comparison is performed.
    """

    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        target_roles: ABCRole | list[ABCRole] | None = None,
        baseline_role: ABCRole | None = None,
        key: Any = "",
        calc_kwargs: dict[str, Any] = {},
    ):
        super().__init__(key=key)
        self.grouping_role = grouping_role or GroupingRole()
        self.target_roles = target_roles or TargetRole()
        self.baseline_role = baseline_role or PreTargetRole()
        self.calc_kwargs = calc_kwargs

    @property
    def search_types(self) -> list[type] | None:
        return None

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
        first_val = next(iter(compare_result.values()))
        if isinstance(first_val, (Dataset, SmallDataset)):
            cr_list_v = list(compare_result.values())
            result = cr_list_v[0]
            if len(cr_list_v) > 1:
                result = result.append(cr_list_v[1:])
            result.index = list(compare_result.keys())
            return result
        return SmallDataset.from_dict(compare_result, roles)


class GroupsComparator(BaseComparator, ABC):
    """
    Comparator that splits data into groups and performs pairwise comparison.
    Supports five comparison modes: groups, columns, columns_in_groups, cross,
    matched_pairs. _inner_function receives raw Dataset slices for each pair.
    """

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
        super().__init__(
            grouping_role=grouping_role,
            target_roles=target_roles,
            baseline_role=baseline_role,
            key=key,
            calc_kwargs=calc_kwargs,
        )
        self.compare_by = compare_by

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

    @staticmethod
    def _check_test_data(test_data: Dataset | None = None) -> Dataset:
        if test_data is None:
            raise ValueError("test_data is needed for evaluation")
        return test_data

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
            bl_dataset = baseline_data[0 if len(baseline_data) == 1 else i][1]
            cmp_dataset = compared_data[i][1]
            
            res_name = (
                compared_data[i][0]
                if compare_by == "groups"
                else f"{compared_data[i][0]}{NAME_BORDER_SYMBOL}{compared_data[i][1].columns[0]}"
            )
            
            if len(bl_dataset) == 0 or len(cmp_dataset) == 0:
                result[res_name] = SmallDataset.from_dict(
                    {
                        "p-value": [np.nan],
                        "statistic": [np.nan],
                        "pass": [False],
                    },
                    StatisticRole(),
                )
                continue
            
            result[res_name] = DatasetAdapter.to_dataset(
                cls._inner_function(
                    bl_dataset,
                    cmp_dataset,
                    **kwargs,
                ),
                InfoRole(),
            )
        return result
    
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
        
        def _safe_slice(ds, cols):
            if cols is None:
                return ds
            if hasattr(ds, '__getitem__'):
                return ds[cols]
            return ds[cols] if isinstance(cols, str) else ds[list(cols)]

        baseline_cols = target_fields if compare_by == "groups" else baseline_field
        baseline_data = [
            (bucket[0], _safe_slice(bucket[1], baseline_cols))
            for bucket in baseline_data
        ]
        compared_data = [
            (bucket[0], _safe_slice(bucket[1], target_fields))
            for bucket in compared_data
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
        
        group_col = group_field_data.columns[0]
        
        if group_col not in target_fields_data.columns:
            target_fields_data = target_fields_data.merge(
                group_field_data, left_index=True, right_index=True, how="left"
            )

        data_buckets = sorted(
            target_fields_data.groupby(by=group_field_data.columns), key=lambda tup: tup[0]
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
        
        group_col = group_field_data.columns[0]
        if group_col not in baseline_field_data.columns:
            baseline_field_data = baseline_field_data.merge(group_field_data, left_index=True, right_index=True, how="left")
        if group_col not in target_fields_data.columns:
            target_fields_data = target_fields_data.merge(group_field_data, left_index=True, right_index=True, how="left")

        baseline_data = baseline_field_data.groupby(by=group_field_data.columns)
        compared_data = cls._split_ds_into_columns(
            target_fields_data.groupby(by=group_field_data.columns)
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
        
        group_col = group_field_data.columns[0]
        if group_col not in baseline_field_data.columns:
            baseline_field_data = baseline_field_data.merge(group_field_data, left_index=True, right_index=True, how="left")
        if group_col not in target_fields_data.columns:
            target_fields_data = target_fields_data.merge(group_field_data, left_index=True, right_index=True, how="left")

        baseline_data = [
            sorted(
                baseline_field_data.groupby(by=group_field_data.columns), key=lambda tup: tup[0]
            ).pop(0)
        ]
        compared_data = sorted(
            target_fields_data.groupby(by=group_field_data.columns), key=lambda tup: tup[0]
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
        
        group_col = group_field_data.columns[0]
        if group_col not in baseline_field_data.columns:
            baseline_field_data = baseline_field_data.merge(group_field_data, left_index=True, right_index=True, how="left")
        if group_col not in target_fields_data.columns:
            target_fields_data = target_fields_data.merge(group_field_data, left_index=True, right_index=True, how="left")

        compared_data = target_fields_data.groupby(by=group_field_data.columns)
        baseline_indexes = baseline_field_data.groupby(by=group_field_data.columns)
        
        baseline_data = []
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
        Splits the given dataset into buckets into baseline and compared data,
        based on the specified comparison mode.

        Args:
            group_field (Union[Sequence[str], str]): The field(s) to group the data by.
            target_fields (Union[str, List[str]]): The field(s) to target for comparison.
            compare_by (Literal['groups', 'columns', 'columns_in_groups', 'cross', 'matched_pairs'], optional):
                The method to compare the data. Defaults to 'groups'.
            baseline_field (Optional[str], optional): The column to use as the baseline for comparison.
                Required if `compare_by` is 'columns' or 'columns_in_groups'. Defaults to None.

        Returns:
            Tuple: A tuple containing the baseline data and the compared data.

        Raises:
            NoRequiredArgumentError: If `baseline_field` is None and `compare_by` is
                'columns' or 'columns_in_groups' or 'cross'.
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
            

    @timeit(level="COMPARATOR", prefix="GROUPS")
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
            if data.ds.tmp_roles:
                return data
            else:
                raise NoColumnsError(TargetRole().role_name)

        if len(group_field_data.columns) != 1 and self.compare_by != "columns":
            raise NotSuitableFieldError(group_field_data, "Grouping")

        if (
            group_field_data.columns[0] in data.groups
        ) and self.compare_by != "matched_pairs":
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
            has_additional = any(
                isinstance(data.additional_fields.roles[col], (AdditionalTargetRole, AdditionalTreatmentRole))
                for col in data.additional_fields.columns
            )
            
            combined_data = (
                data.ds.merge(
                    data.additional_fields[
                        [
                            col
                            for col in data.additional_fields.columns
                            if isinstance(
                                data.additional_fields.roles[col], (AdditionalTargetRole, AdditionalTreatmentRole)
                            )
                        ]
                    ],
                    left_index=True,
                    right_index=True,
                    how="outer",
                )
                if has_additional
                else data.ds
            )
            
            group_col_name = group_field_data.columns[0]
            if group_col_name in combined_data.columns:
                inner_df = combined_data.data if hasattr(combined_data, 'data') else combined_data.backend_data.data
                initial_len = len(inner_df)
                inner_df = inner_df.dropna(subset=[group_col_name])
                dropped = initial_len - len(inner_df)
                if dropped > 0:
                    combined_data = type(combined_data)(
                        data=inner_df,
                        roles={c: combined_data.roles.get(c, InfoRole()) for c in inner_df.columns},
                    )
            
            data.groups[group_field_data.columns[0]] = {
                f"{group}": ds for group, ds in combined_data.groupby(group_field_data.columns[0])
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
        )
        result_dataset = self._local_extract_dataset(
            compare_result, {key: StatisticRole() for key in compare_result}
        )
        return self._set_value(data, result_dataset)


# Backward-compatible alias — existing code importing Comparator continues to work.
Comparator = GroupsComparator


class GroupHypothesisTesting(GroupsComparator, ABC):
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


class StatsComparator(BaseComparator, ABC):
    """Two-phase comparator that operates on aggregated statistics instead of raw data.
    
    Phase 1 — Aggregate: `_compute_stats()` delegates to StatsAggregationExtension.
    Phase 2 — Compare: `_inner_function()` receives the per-column stats dicts of two
    groups (baseline vs compared) and returns the test result for that column.
    """
    STAT_FUNCTIONS: ClassVar[dict[str, Callable[[Dataset], Any]]] = {
        "mean": lambda d: d.mean(),
        "var": lambda d: d.var(),
        "std": lambda d: d.std(),
        "count": lambda d: len(d),
        "sum": lambda d: d.sum(),
        "min": lambda d: d.min(),
        "max": lambda d: d.max(),
    }
    
    REQUIRED_STATS: ClassVar[list[str]] = []

    def __init__(
        self,
        stats: list[str],
        grouping_role: ABCRole | None = None,
        target_roles: ABCRole | list[ABCRole] | None = None,
        key: Any = "",
        calc_kwargs: dict[str, Any] = {},
    ):
        super().__init__(
            grouping_role=grouping_role,
            target_roles=target_roles,
            key=key,
            calc_kwargs=calc_kwargs,
        )
        self.stats = stats

    @classmethod
    @timeit(level="AGG", prefix="STATS")
    def _compute_stats(
        cls,
        data: Dataset,
        group_cols: list[str],
        target_columns: list[str],
        stats: list[str] | None = None,
        **kwargs,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Computes aggregated statistics for the specified groups and columns.
        
        Args:
            data: The dataset to aggregate.
            group_cols: List of column names to group by.
            target_columns: List of target column names to compute stats for.
            stats: List of statistical functions to apply. Defaults to REQUIRED_STATS.
            **kwargs: Additional arguments for the aggregation extension.
            
        Returns:
            A nested dictionary mapping group names to target columns to stat values.
        """
        from ..extensions.stats_hypothesis_testing import StatsAggregationExtension
        
        stats = stats or cls.REQUIRED_STATS
        ext = StatsAggregationExtension()
        return ext.calc(
            data=data,
            group_cols=group_cols,
            target_cols=target_columns,
            stats=stats,
        )

    @classmethod
    @abstractmethod
    def _inner_function(
        cls,
        baseline_stats: dict[str, Any],
        compared_stats: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        """Computes the comparison metric using aggregated statistics.
        
        Args:
            baseline_stats: Dictionary of aggregated stats for the baseline group.
            compared_stats: Dictionary of aggregated stats for the compared group.
            **kwargs: Additional keyword arguments for the specific test.
            
        Returns:
            A dictionary containing the computed test results (e.g., p-value, statistic).
            
        Raises:
            AbstractMethodError: If not implemented by a subclass.
        """
        raise AbstractMethodError

    def _set_stats_value(self, data: ExperimentData, value: Dataset) -> ExperimentData:
        data.set_value(
            ExperimentDataEnum.analysis_tables,
            f"{self.id}{NAME_BORDER_SYMBOL}stats",
            value,
        )
        return data

    @timeit(level="COMPARATOR", prefix="STATS")
    def execute(self, data: ExperimentData) -> ExperimentData:
        fields = self._get_fields_data(data)
        group_field_data = fields["group_field"]
        target_fields_data = fields["target_fields"]

        if len(target_fields_data.columns) == 0:
            if data.ds.tmp_roles:
                return data
            raise NoColumnsError(TargetRole().role_name)

        if len(group_field_data.columns) != 1:
            raise NotSuitableFieldError(group_field_data, "Grouping")

        all_cols = list(target_fields_data.columns) + [
            c for c in group_field_data.columns if c not in target_fields_data.columns
        ]
        merged_data = data.ds[all_cols]
        
        group_col_stats = self._compute_stats(
            data=merged_data,
            group_cols=list(group_field_data.columns),
            target_columns=list(target_fields_data.columns),
            stats=self.stats,
        )
        
        group_names = list(group_col_stats.keys())

        if len(group_names) < 2:
            for col in target_fields_data.columns:
                self.key = str(col)
                self._set_value(data, SmallDataset.create_empty())
            return data

        for col in target_fields_data.columns:
            self.key = str(col) 

            stats_data_dict = {}
            for stat in self.stats:
                stats_data_dict[f"{stat}{NAME_BORDER_SYMBOL}{col}"] = [
                    group_col_stats[g][col][stat] for g in group_names
                ]
            
            stats_dataset = DatasetAdapter.to_dataset(stats_data_dict, StatisticRole())
            stats_dataset.index = group_names
            data = self._set_stats_value(data, stats_dataset)

            baseline_name = group_names[0]
            col_results = []
            for compared_name in group_names[1:]:
                res = self._inner_function(
                    group_col_stats[baseline_name][col],
                    group_col_stats[compared_name][col],
                    **self.calc_kwargs,
                )
                col_results.append(DatasetAdapter.to_dataset(res, StatisticRole()))
            
            if col_results:
                result_dataset = col_results[0].append(col_results[1:])
                result_dataset.index = [str(g) for g in group_names[1:]]
                data = self._set_value(data, result_dataset)
                
        self.key = str(
            target_fields_data.columns[0] 
            if len(target_fields_data.columns) == 1 
            else list(target_fields_data.columns)
        )
        return data


class StatsHypothesisTesting(StatsComparator, ABC):
    """
    StatsComparator subclass that adds a ``reliability`` parameter — the direct
    analog of :class:`GroupHypothesisTesting` for the stats-based comparator branch.

    Concrete subclasses only need to implement ``_inner_function`` and declare
    ``REQUIRED_STATS``; the ``reliability`` value is forwarded via ``calc_kwargs``
    so it is available inside ``_inner_function`` without extra wiring.
    """

    def __init__(
        self,
        stats: list[str],
        grouping_role: ABCRole | None = None,
        target_roles: ABCRole | list[ABCRole] | None = None,
        reliability: float = 0.05,
        key: Any = "",
        calc_kwargs: dict[str, Any] = {},
    ):
        merged_kwargs = {"reliability": reliability, **calc_kwargs}
        super().__init__(
            stats=stats,
            grouping_role=grouping_role,
            target_roles=target_roles,
            key=key,
            calc_kwargs=merged_kwargs,
        )
        self.reliability = reliability
        
        
    
class AdaptiveHypothesisTest(BaseComparator):
    """
    Routes execute() to a backend-specific hypothesis test at runtime.

    Subclasses declare BACKEND_MAP: {BackendsEnum -> concrete test class}.
    The concrete class must be a subclass of either GroupHypothesisTesting or
    StatsHypothesisTesting. Results are always stored under the adaptive
    instance's own id, so pipeline lookups remain consistent regardless of
    which backend is active.
    """

    BACKEND_MAP: ClassVar[dict[BackendsEnum, type[BaseComparator]]] = {}

    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        target_roles: ABCRole | None = None,
        reliability: float = 0.05,
        compare_by: Literal[
            "groups", "columns", "columns_in_groups", "cross", "matched_pairs"
        ] = "groups",
        key: Any = "",
    ):
        super().__init__(grouping_role=grouping_role, target_roles=target_roles, key=key)
        self.reliability = reliability
        self.compare_by = compare_by

    def _build_delegate(self, cls: type[BaseComparator]) -> BaseComparator:
        """
        Instantiate *cls* with this instance's configuration and override its
        id to match ours so results land under the adaptive class's id.
        """
        if issubclass(cls, StatsHypothesisTesting):
            instance = cls(
                grouping_role=self.grouping_role,
                target_roles=self.target_roles,
                reliability=self.reliability,
                key=self.key,
            )
        else:  # GroupHypothesisTesting branch
            instance = cls(
                compare_by=self.compare_by,
                grouping_role=self.grouping_role,
                target_role=self.target_roles,
                reliability=self.reliability,
                key=self.key,
            )
        # Ensure analysis_tables entries use our id, not the delegate's.
        instance._id = self._id
        return instance
    
    @staticmethod
    def _inner_function(data, **kwargs):
        raise NotImplementedError("Use execute() for adaptive tests.")

    def execute(self, data: ExperimentData) -> ExperimentData:
        backend = data.ds.backend_type
        if backend not in self.BACKEND_MAP:
            raise ValueError(
                f"{type(self).__name__} has no implementation for backend {backend!r}. "
                f"Registered: {list(self.BACKEND_MAP)}"
            )
        return self._build_delegate(self.BACKEND_MAP[backend]).execute(data)
