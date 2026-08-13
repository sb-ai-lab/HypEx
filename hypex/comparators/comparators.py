from __future__ import annotations

from typing import Literal, Any

import numpy as np

from ..dataset import ABCRole, Dataset
from ..utils.constants import NUMBER_TYPES_LIST
from .abstract import Comparator, BaseComparator, StatsComparator
NUM_OF_BUCKETS = 10


class GroupDifference(StatsComparator):
    """Comparator for calculating the difference in means between groups.

    Computes absolute and percentage differences between baseline and compared
    group means using pre-aggregated statistics. Inherits from StatsComparator
    to leverage vectorized Spark aggregation instead of iterative raw-data
    processing, significantly reducing execution time and DAG lineage size.

    Attributes:
        REQUIRED_STATS: List of statistics required for computation (["mean"]).
        compare_by: Comparison mode identifier. Retained for backward
            compatibility with the previous GroupsComparator interface.

    Example:
        >>> diff = GroupDifference(grouping_role=TreatmentRole())
        >>> result = diff.execute(experiment_data)
    """

    REQUIRED_STATS = ["mean"]

    def __init__(
        self,
        compare_by: Literal[
            "groups", "columns", "columns_in_groups", "cross", "matched_pairs"
        ] = "groups",
        grouping_role: ABCRole | None = None,
        target_roles: ABCRole | list[ABCRole] | None = None,
    ):
        """Initializes the GroupDifference comparator.

        Args:
            compare_by: Comparison mode. Only "groups" is functionally used
                by StatsComparator; other values are kept for API compatibility.
                Defaults to "groups".
            grouping_role: Role identifying the column to split data into
                baseline and treatment groups. Defaults to GroupingRole().
            target_roles: Role(s) identifying numeric columns for which to
                compute mean differences. Defaults to TargetRole().
        """
        super().__init__(
            stats=self.REQUIRED_STATS,
            grouping_role=grouping_role,
            target_roles=target_roles,
        )
        self.compare_by = compare_by

    @property
    def search_types(self) -> list[type] | None:
        """Returns the list of data types eligible for mean comparison.

        Returns:
            List containing int and float types, as mean difference
            is only meaningful for numeric columns.
        """
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls,
        baseline_stats: dict[str, Any],
        compared_stats: dict[str, Any],
        **kwargs,
    ) -> dict:
        """Computes mean difference metrics from pre-aggregated statistics.

        Calculates the absolute difference and percentage change between
        baseline and compared group means. Handles edge cases where means
        are missing or the baseline mean is zero.

        Args:
            baseline_stats: Aggregated statistics dict for the baseline group.
                Must contain a "mean" key.
            compared_stats: Aggregated statistics dict for the compared group.
                Must contain a "mean" key.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dictionary with keys:
                - "control mean": Baseline group mean.
                - "test mean": Compared group mean.
                - "difference": Absolute difference (test - control).
                - "difference %": Percentage change relative to control.
                  Returns None if control mean is zero or either mean is missing.
        """
        control_mean = baseline_stats.get("mean")
        test_mean = compared_stats.get("mean")

        if control_mean is None or test_mean is None:
            return {
                "control mean": control_mean,
                "test mean": test_mean,
                "difference": None,
                "difference %": None,
            }

        difference = test_mean - control_mean
        difference_pct = (
            (test_mean / control_mean - 1) * 100 if control_mean != 0 else None
        )

        return {
            "control mean": control_mean,
            "test mean": test_mean,
            "difference": difference,
            "difference %": difference_pct,
        }


class GroupSizes(StatsComparator):
    """Comparator for calculating group sizes and their proportions.

    Computes absolute counts and percentage shares for baseline and compared
    groups using pre-aggregated count statistics. Uses the grouping column
    itself as the target to avoid unnecessary scanning of all target columns.

    Attributes:
        REQUIRED_STATS: List of statistics required for computation (["count"]).
        compare_by: Comparison mode identifier. Retained for backward
            compatibility with the previous GroupsComparator interface.

    Example:
        >>> sizes = GroupSizes(grouping_role=TreatmentRole())
        >>> result = sizes.execute(experiment_data)
    """

    REQUIRED_STATS = ["count"]

    def __init__(
        self,
        compare_by: Literal[
            "groups", "columns", "columns_in_groups", "cross", "matched_pairs"
        ] = "groups",
        grouping_role: ABCRole | None = None,
    ):
        """Initializes the GroupSizes comparator.

        Sets target_roles to grouping_role so that only the grouping column
        is scanned for count aggregation, avoiding redundant processing of
        all target columns.

        Args:
            compare_by: Comparison mode. Only "groups" is functionally used
                by StatsComparator; other values are kept for API compatibility.
                Defaults to "groups".
            grouping_role: Role identifying the column defining group
                membership. Also used as the target for count aggregation.
                Defaults to GroupingRole().
        """
        super().__init__(
            stats=self.REQUIRED_STATS,
            grouping_role=grouping_role,
            target_roles=grouping_role,
        )
        self.compare_by = compare_by

    @classmethod
    def _inner_function(
        cls,
        baseline_stats: dict[str, Any],
        compared_stats: dict[str, Any],
        **kwargs,
    ) -> dict:
        """Computes group size metrics from pre-aggregated count statistics.

        Calculates absolute sizes and percentage shares for baseline and
        compared groups. Returns 0.0 for percentages when total size is zero.

        Args:
            baseline_stats: Aggregated statistics dict for the baseline group.
                Must contain a "count" key.
            compared_stats: Aggregated statistics dict for the compared group.
                Must contain a "count" key.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dictionary with keys:
                - "control size": Number of observations in the baseline group.
                - "test size": Number of observations in the compared group.
                - "control size %": Baseline share of total observations.
                - "test size %": Compared share of total observations.
        """
        size_a = baseline_stats.get("count", 0)
        size_b = compared_stats.get("count", 0)
        total = size_a + size_b

        return {
            "control size": size_a,
            "test size": size_b,
            "control size %": (size_a / total) * 100 if total > 0 else 0.0,
            "test size %": (size_b / total) * 100 if total > 0 else 0.0,
        }


class PSI(Comparator):
    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> dict[str, float]:
        test_data = cls._check_test_data(test_data=test_data)
        data.sort(ascending=False)
        test_data.sort(ascending=False)
        data_column = data.iloc[:, 0]
        test_data_column = test_data.iloc[:, 0]
        data_bins = np.arange(
            data_column.min(),
            data_column.max(),
            (data_column.max() - data_column.min()) / NUM_OF_BUCKETS,
        )
        test_data_bins = np.arange(
            test_data_column.min(),
            test_data_column.max(),
            (test_data_column.max() - test_data_column.min()) / NUM_OF_BUCKETS,
        )
        data_groups = data_column.groupby(
            data_column.cut(data_bins).get_values(column=data.columns[0])
        )
        test_data_groups = test_data_column.groupby(
            test_data_column.cut(test_data_bins).get_values(column=test_data.columns[0])
        )

        data_psi = [x[1].count() / len(data) for x in data_groups]
        test_data_psi = [x[1].count() / len(test_data) for x in test_data_groups]
        psi = [(y - x) * np.log(y / x) for x, y in zip(data_psi, test_data_psi)]
        return {"PSI": sum(psi)}

class StatTestMasterAbstract(BaseComparator):
    """
    Master-abstract class for stat-tests
    """
    # def __init__(self, **experiment_kwargs):
    #     self._experiment_kwargs: dict[str, Any] = experiment_kwargs
    def __init__(
            self,
            grouping_role: ABCRole | None = None,
            target_roles: ABCRole | None = None,
            baseline_role: ABCRole | None = None,
            reliability: float = 0.05,
            compare_by: Literal[
                "groups", "columns", "columns_in_groups", "cross", "matched_pairs"
            ] = "groups",
            key: Any = "",
    ):
        super().__init__(grouping_role=grouping_role, target_roles=target_roles, baseline_role=baseline_role, key=key)
        self.reliability = reliability
        self.compare_by = compare_by

    @staticmethod
    def _inner_function(data, **kwargs):
        pass

    def execute(self, data):
        pass

    @property
    def experiment_kwargs(self):
        return self._experiment_kwargs

# Master-backend classes for stat-tests
class TTest(StatTestMasterAbstract):
    """
    T-test master-backend class.
    """

class Chi2Test(StatTestMasterAbstract):
    """
    Chi-square test master-backend class.
    """

class KSTest(StatTestMasterAbstract):
    """
    KS-test master-backend class.
    """

class UTest(StatTestMasterAbstract):
    """
    KS-test master-backend class.
    """

class ZTest(StatTestMasterAbstract):
    """
    Z-test masker-backend class.
    """