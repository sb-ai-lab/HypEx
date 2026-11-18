from __future__ import annotations

import math
from typing import Any, ClassVar, Literal

import numpy as np

from ..dataset import ABCRole, Dataset
from ..utils.constants import NUMBER_TYPES_LIST
from .abstract import BaseComparator, Comparator, StatsComparator

NUM_OF_BUCKETS = 10


class GroupDifference(StatsComparator):
    """Comparator for calculating the difference in means between groups.

    Computes absolute and percentage differences between baseline and compared
    group means using pre-aggregated statistics. Inherits from StatsComparator
    to leverage vectorized Spark aggregation instead of iterative raw-data
    processing, significantly reducing execution time and DAG lineage size.

    Computes absolute and percentage differences between baseline and compared
    group means, along with the 95% Confidence Interval (CI) for the absolute
    difference. Uses pre-aggregated statistics (mean, variance, count) to
    leverage vectorized Spark aggregation.

    Attributes:
        REQUIRED_STATS: List of statistics required for computation (["mean"]).
        compare_by: Comparison mode identifier. Retained for backward
            compatibility with the previous GroupsComparator interface.

    Example:
        >>> diff = GroupDifference(grouping_role=TreatmentRole())
        >>> result = diff.execute(experiment_data)
    """

    REQUIRED_STATS: ClassVar[list[str]] = ["mean", "var", "count"]
    # Z-score for 95% confidence level
    Z_SCORE_95: ClassVar[float] = 1.96

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
            compare_by=compare_by,
            grouping_role=grouping_role,
            target_roles=target_roles,
        )

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
    ) -> dict[str, Any]:
        """Computes mean difference metrics and 95% Confidence Interval.

        Calculates the absolute difference, percentage change, and the 95% CI
        bounds using the standard error of the difference between two independent
        means.

        Args:
            baseline_stats: Aggregated statistics dict for the baseline group.
                Must contain "mean", "var", and "count".
            compared_stats: Aggregated statistics dict for the compared group.
                Must contain "mean", "var", and "count".
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dictionary with keys: "control mean", "test mean", "difference",
            "difference %", "ci lower", "ci upper".
        """
        control_mean = baseline_stats.get("mean")
        test_mean = compared_stats.get("mean")

        if control_mean is None or test_mean is None:
            return {
                "control mean": control_mean,
                "test mean": test_mean,
                "difference": None,
                "difference %": None,
                "ci lower": None,
                "ci upper": None,
            }

        difference = test_mean - control_mean
        difference_pct = (
            (test_mean / control_mean - 1) * 100 if control_mean != 0 else None
        )

        # --- Confidence Interval Calculation ---
        ci_lower = None
        ci_upper = None

        control_var = baseline_stats.get("var")
        test_var = compared_stats.get("var")
        control_n = baseline_stats.get("count")
        test_n = compared_stats.get("count")

        # Check if we have enough data to calculate Standard Error
        if (
            control_var is not None
            and test_var is not None
            and control_n is not None
            and test_n is not None
            and control_n > 1
            and test_n > 1
        ):
            # Standard Error (SE) of the difference between two independent means
            se = math.sqrt((control_var / control_n) + (test_var / test_n))

            ci_lower = difference - (cls.Z_SCORE_95 * se)
            ci_upper = difference + (cls.Z_SCORE_95 * se)

        return {
            "control mean": control_mean,
            "test mean": test_mean,
            "difference": difference,
            "difference %": difference_pct,
            "ci lower": ci_lower,
            "ci upper": ci_upper,
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

    REQUIRED_STATS: ClassVar[list[str]] = ["count"]

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
            compare_by=compare_by,
            grouping_role=grouping_role,
            target_roles=grouping_role,
        )

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
        if len(data) == 0 or len(test_data) == 0:
            raise ValueError("PSI requires non-empty data and test_data")
        data_column = data.iloc[:, 0]
        test_data_column = test_data.iloc[:, 0]
        lower = min(float(data_column.min()), float(test_data_column.min()))
        upper = max(float(data_column.max()), float(test_data_column.max()))
        if lower == upper:
            return {
                "PSI": 0.0
                if float(data_column.min()) == float(test_data_column.min())
                else float("inf")
            }

        edges = np.linspace(lower, upper, NUM_OF_BUCKETS + 1)
        epsilon = np.finfo(float).eps
        psi = 0.0
        for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            data_bin = data_column >= left
            test_bin = test_data_column >= left
            if i > 0:
                data_bin = data_column > left
                test_bin = test_data_column > left
            if i == NUM_OF_BUCKETS - 1:
                data_bin = data_bin & (data_column <= right)
                test_bin = test_bin & (test_data_column <= right)
            else:
                data_bin = data_bin & (data_column < right)
                test_bin = test_bin & (test_data_column < right)
            data_count = data_bin.astype({data_bin.columns[0]: int}).sum()
            test_count = test_bin.astype({test_bin.columns[0]: int}).sum()
            data_ratio = max(float(data_count) / len(data), epsilon)
            test_ratio = max(float(test_count) / len(test_data), epsilon)
            psi += (test_ratio - data_ratio) * np.log(test_ratio / data_ratio)
        return {"PSI": psi}


class StatTestMasterAbstract(BaseComparator):
    """
    Master-abstract class for stat-tests
    """

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
        super().__init__(
            grouping_role=grouping_role,
            target_roles=target_roles,
            baseline_role=baseline_role,
            key=key,
        )
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

    On Spark the p-value is a binned approximation (see ``StatsKSTest``).
    """


class UTest(StatTestMasterAbstract):
    """
    U-test (Mann-Whitney) master-backend class.

    On Spark the p-value is a binned approximation (see ``StatsUTest``).
    """


class ZTest(StatTestMasterAbstract):
    """
    Z-test masker-backend class.
    """
