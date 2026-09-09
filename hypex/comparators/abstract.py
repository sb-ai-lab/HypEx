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
    GroupedDataset,
    GroupingRole,
    InfoRole,
    PreTargetRole,
    SmallDataset,
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
    timeit,
)
from ..utils.errors import (
    AbstractMethodError,
    NoColumnsError,
    NoRequiredArgumentError,
    NotSuitableFieldError,
)


class BaseComparator(Calculator):
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
        def _safe_slice(ds, cols):
            if cols is None:
                return ds
            if hasattr(ds, '__getitem__'):
                return ds[cols]
            return ds[cols] if isinstance(cols, str) else ds[list(cols)]

        if not isinstance(grouping_data, dict):
            raise TypeError(
                f"Grouping data must be dict of strings and datasets, but got {type(grouping_data)}"
            )
        compared_data = sorted(grouping_data.items(), key=lambda x: str(x[0]))
        baseline_data = [compared_data.pop(0)]

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
            target_fields_data.groupby(by=group_field_data.columns),
            key=lambda tup: tup[0],
        )

        target_cols = [c for c in target_fields_data.columns if c != group_col]

        baseline_bucket = data_buckets.pop(0)
        baseline_data = cls._split_ds_into_columns(
            [(baseline_bucket[0], baseline_bucket[1][target_cols])]
        )
        compared_data = cls._split_ds_into_columns(
            [(key, ds[target_cols]) for key, ds in data_buckets]
        )
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

        baseline_indexes = baseline_field_data.merge(group_field_data, left_index=True, right_index=True).groupby(by=group_field_data.columns[0])
        baseline_data = []
        compared_data = []
        for group, indexes in group_field_data.reset_index().groupby(group_field_data.columns[0]):
            compared_data.append((group, target_fields_data.loc[indexes.iget_values(column=0), :]))

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
            combined_data = data.ds

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
    """
    Two-phase comparator that operates on aggregated statistics instead of raw data.

    Phase 1 — Aggregate: _compute_stats() is called once per group with the full
    multi-column group slice. It returns {col: {stat: value}} for all target columns
    in a single pass, allowing backends (e.g. Spark) to issue one aggregation job
    instead of one per column.

    Phase 2 — Compare: _inner_function() receives the per-column stats dicts of two
    groups (baseline vs compared) and returns the test result for that column.

    Two datasets are stored in analysis_tables:
    - ``{self.id}{NAME_BORDER_SYMBOL}stats`` — per-group stats table (rows=groups,
      cols={stat}{NAME_BORDER_SYMBOL}{col})
    - ``self.id`` — pairwise test results in the same shape as GroupsComparator output

    This design is particularly efficient for Spark backends, where Phase 1 runs
    as distributed aggregations and only small scalar dicts reach the driver.
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

    # The statistics this comparator needs; concrete subclasses override it.
    REQUIRED_STATS: ClassVar[list[str]] = []

    def __init__(
        self,
        stats: list[str],
        compare_by: Literal["groups", "matched_pairs"],
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
        self.stats = stats
        self.compare_by = compare_by

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
            grouped: GroupedDataset (result of target_fields_data.groupby(by=group_field_data)).
            stats: List of stat names to compute (e.g. ["mean", "var", "count"]).

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

    @classmethod
    def calc(
        cls,
        target_fields_data: Dataset | None = None,
        group_field_data: Dataset | None = None,
        baseline_fields_data: Dataset | None = None,
        stats: list[str] | None = None,
        compare_by: str = "groups",
        group_col_stats: dict[str, dict[str, dict[str, Any]]] | None = None,
        **kwargs,
    ) -> dict:
        """
        Stateless entry point mirroring :meth:`GroupsComparator.calc`, so the
        comparator can be run outside the experiment pipeline.

        Pass either pre-aggregated ``group_col_stats`` (as produced by
        :meth:`_compute_stats`) or the raw ``target_fields_data`` and
        ``group_field_data`` to have the statistics aggregated here. ``stats``
        defaults to the comparator's ``REQUIRED_STATS``, so callers normally
        don't need to supply it.

        Returns ``{f"{group}{NAME_BORDER_SYMBOL}{col}": Dataset}`` pairwise test
        results, comparing every non-baseline group against the first group.
        """
        if group_col_stats is None:
            if target_fields_data is None or group_field_data is None:
                raise ValueError(
                    "You should pass either group_col_stats or both "
                    "target_fields_data and group_field_data."
                )

            grouped = cls._prepare_data(compare_by, target_fields_data, group_field_data, baseline_fields_data)
            group_col_stats = cls._compute_stats(
                grouped, list(target_fields_data.columns), stats or cls.REQUIRED_STATS
            )

        return cls._execute_inner_function(
            group_col_stats=group_col_stats,
            compare_by=compare_by,
            **kwargs
        )

    @classmethod
    def _execute_inner_function(
        cls,
        group_col_stats: dict[str, dict[str, dict[str, Any]]],
        compare_by: str,
        **kwargs
    ) -> list[Dataset | SmallDataset]:
        group_names = list(group_col_stats.keys())
        if len(group_names) < 2:
            return []

        baseline_name = group_names[0]
        if compare_by == "groups":
            result_ds_list = [
                DatasetAdapter.to_dataset(
                    cls._inner_function(
                        group_col_stats[baseline_name][col],
                        group_col_stats[compared_name][col],
                        **kwargs,
                    ),
                    StatisticRole(),
                )
                for compared_name in group_names[1:]
                for col in  group_col_stats[baseline_name]
            ]
        elif compare_by == "matched_pairs":
            result_ds_list = [
                DatasetAdapter.to_dataset(
                    cls._inner_function(
                        group_col_stats[groups_name][col],
                        group_col_stats[groups_name][col + "_matched"],
                        **kwargs,
                    ),
                    StatisticRole(),
                )
                for groups_name in group_names
                for col in group_col_stats[baseline_name] if not col.endswith('_matched')
            ]
        return result_ds_list

    @staticmethod
    def _prepare_data(
        compare_by: str,
        target_fields_data: Dataset | None = None,
        group_field_data: Dataset | None = None,
        baseline_fields_data: Dataset | None = None,
    ) -> GroupedDataset:
        if compare_by == "groups":
            group_col = group_field_data.columns[0]
            if group_col in target_fields_data.columns:
                grouped: GroupedDataset = target_fields_data.groupby(
                    by=group_field_data.columns
                )
            else:
                grouped: GroupedDataset = (
                    target_fields_data
                    .merge(group_field_data, left_index=True, right_index=True)
                    .groupby(by=group_field_data.columns)
                )
        elif compare_by == "matched_pairs":
            best_match_col = baseline_fields_data.columns[0]
            group_col = group_field_data.columns[0]
            baseline_fields = baseline_fields_data[best_match_col]
            tmp_data = group_field_data.merge(
                right=target_fields_data, left_index=True, right_index=True
            )
            tmp_data = tmp_data.merge(
                right=baseline_fields, right_index=True, left_index=True
            )
            tmp_data = tmp_data.merge(
                right=tmp_data, right_index=True, left_on=best_match_col,
                suffixes=("", "_matched"),
            )
            prepeared_data = tmp_data.drop(
                columns=[best_match_col, best_match_col + "_matched", group_col + "_matched"]
            )
            grouped: GroupedDataset = prepeared_data.groupby(by=group_col)
        return grouped

    @timeit(level="COMPARATOR", prefix="STATS")
    def execute(self, data: ExperimentData) -> ExperimentData:
        """Execute the stats-based comparator on the given experiment data.

        The execution follows a two-phase design optimised for distributed backends:

        **Phase 1 - Aggregate:**
            A single aggregation call computes all required statistics for every
            target column across all groups in ONE backend job (e.g. one Spark
            ``groupBy().agg()``).  This avoids the NxM job explosion that would
            occur if each (group, column) pair were aggregated separately.

        **Phase 2 - Compare:**
            The pre-aggregated statistics are fed pairwise (baseline vs. each
            compared group) into ``_inner_function``, which returns the test
            result (p-value, statistic, pass) for each (group, column) pair.

        Two artifacts are stored in ``analysis_tables``:

        * ``{self.id}┆stats`` - per-group statistics table
        (rows = groups, cols = ``{stat}┆{col}``).
        * ``{self.id}`` - pairwise test results in the same shape as
        ``GroupsComparator`` output.

        Comparison modes
        ----------------
        * ``"groups"`` - standard multi-group comparison.  The first
        (alphabetically smallest) group is treated as baseline.
        * ``"matched_pairs"`` - each observation is compared against its
        matched counterpart.  Requires ``baseline_fields_data`` (e.g.
        ``AdditionalMatchingRole``) containing match indices.

        Args:
            data: The ``ExperimentData`` container holding the datasets,
                roles, and any previously computed analysis tables.

        Returns:
            The updated ``ExperimentData`` with test results stored in
            ``analysis_tables`` under this executor's ``self.id``.

        Raises:
            NoColumnsError: If no target columns are found and no temporary
                roles are active.
            NotSuitableFieldError: If the grouping field has more than one
                column or fewer than two groups are detected.
        """
        # ── 1. Resolve field subsets from roles ──────────────────────────────
        fields = self._get_fields_data(data)
        group_field_data = fields["group_field"]
        target_fields_data = fields["target_fields"]
        baseline_fields_data = fields["baseline_field"]

        # ── 2. Early-exit / validation guards ────────────────────────────────
        if len(target_fields_data.columns) == 0:
            # When tmp_roles are active an empty target set simply means
            # "no column matched this iteration" – not an error.
            if data.ds.tmp_roles:
                return data
            raise NoColumnsError(TargetRole().role_name)

        if len(group_field_data.columns) != 1:
            raise NotSuitableFieldError(group_field_data, "Grouping")

        # Set the composite key used for result storage / pipeline lookups.
        self.key = str(
            target_fields_data.columns[0]
            if len(target_fields_data.columns) == 1
            else list(target_fields_data.columns)
        )

        group_col = group_field_data.columns[0]
        target_cols = list(target_fields_data.columns)

        # ── 3. Phase 1 – Aggregate all stats in a single backend job ─────────
        if self.compare_by == "groups":
            # Build a column projection that contains the group column and all
            # target columns.  This avoids a merge (which can duplicate column
            # names when target_roles == grouping_role, e.g. GroupSizes).
            all_cols = target_cols + (
                [group_col] if group_col not in target_cols else []
            )
            agg_data = data.ds[all_cols]

            group_col_stats = self._compute_stats(
                data=agg_data,
                group_cols=[group_col],
                target_columns=target_cols,
                grouped=None,          # unused by StatsAggregationExtension
                stats=self.stats,
            )

        elif self.compare_by == "matched_pairs":
            # For matched pairs we must first build a dataset that contains
            # both the original values and the matched (counterfactual) values
            # side-by-side, then aggregate both sets.
            best_match_col = baseline_fields_data.columns[0]
            baseline_fields = baseline_fields_data[best_match_col]

            # Merge: group_col + target columns + match-index column
            tmp_data = group_field_data.merge(
                right=target_fields_data, left_index=True, right_index=True
            )
            tmp_data = tmp_data.merge(
                right=baseline_fields, right_index=True, left_index=True
            )
            # Self-merge on the match-index column to pull in the matched rows.
            # Original columns keep their names; matched counterparts get
            # the "_matched" suffix.
            tmp_data = tmp_data.merge(
                right=tmp_data,
                right_index=True,
                left_on=best_match_col,
                suffixes=("", "_matched"),
            )
            # Drop auxiliary columns that are not part of the comparison.
            prepared_data = tmp_data.drop(
                columns=[
                    best_match_col,
                    best_match_col + "_matched",
                    group_col + "_matched",
                ]
            )

            # The target columns for aggregation include both original and
            # matched variants so that _inner_function can compare them.
            matched_target_cols = target_cols + [
                f"{c}_matched" for c in target_cols
            ]

            group_col_stats = self._compute_stats(
                data=prepared_data,
                group_cols=[group_col],
                target_columns=matched_target_cols,
                grouped=None,
                stats=self.stats,
            )
        else:
            raise ValueError(
                f"StatsComparator supports 'groups' and 'matched_pairs' only, "
                f"got compare_by={self.compare_by!r}"
            )

        # ── 4. Validate that at least two groups exist ───────────────────────
        group_names = sorted(group_col_stats.keys(), key=str)

        if len(group_names) < 2:
            # Fewer than two groups → nothing to compare.  Store empty results
            # so downstream reporters do not crash on a missing key.
            for col in target_cols:
                self.key = str(col)
                self._set_value(data, SmallDataset.create_empty())
            return data

        # ── 5. Store per-column statistics tables ────────────────────────────
        # Each column gets a small table (rows = groups, cols = stats) stored
        # under "{self.id}┆stats".  This is useful for debugging and for
        # reporters that display raw aggregates.
        for col in target_cols:
            self.key = str(col)
            stats_data_dict: dict[str, list] = {}
            for stat in self.stats:
                stats_data_dict[f"{stat}{NAME_BORDER_SYMBOL}{col}"] = [
                    group_col_stats[g][col][stat] for g in group_names
                ]
            stats_dataset = DatasetAdapter.to_dataset(stats_data_dict, StatisticRole())
            stats_dataset.index = group_names
            data = self._set_stats_value(data, stats_dataset)

        # ── 6. Phase 2 – Pairwise comparison via _inner_function ─────────────
        result_ds_list = self._execute_inner_function(
            group_col_stats=group_col_stats,
            compare_by=self.compare_by,
            **self.calc_kwargs,
        )

        if not result_ds_list:
            return data

        # ── 7. Assemble the final result Dataset ─────────────────────────────
        result_dataset = result_ds_list[0].append(result_ds_list[1:])

        if self.compare_by == "groups":
            # Index: one row per (compared_group, target_column) pair.
            result_dataset.index = [
                f"{compared_name}{NAME_BORDER_SYMBOL}{col}"
                for compared_name in group_names[1:]
                for col in target_cols
            ]
        elif self.compare_by == "matched_pairs":
            # Index: one row per (group, target_column) pair.
            result_dataset.index = [
                f"{group_name}{NAME_BORDER_SYMBOL}{col}"
                for group_name in group_names
                for col in target_cols
            ]

        # Restore the composite key before storing the final result.
        self.key = str(
            target_cols[0] if len(target_cols) == 1 else target_cols
        )
        return self._set_value(data, result_dataset)


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
        compare_by: Literal["groups", "matched_pairs"],
        grouping_role: ABCRole | None = None,
        target_roles: ABCRole | list[ABCRole] | None = None,
        baseline_role: ABCRole | None = None,
        reliability: float = 0.05,
        key: Any = "",
        calc_kwargs: dict[str, Any] = {},
    ):
        merged_kwargs = {"reliability": reliability, **calc_kwargs}
        super().__init__(
            stats=stats,
            compare_by=compare_by,
            grouping_role=grouping_role,
            target_roles=target_roles,
            baseline_role=baseline_role,
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
