from __future__ import annotations

import math
from math import sqrt
from typing import Any, ClassVar, Literal

import numpy as np
from scipy.stats import chi2_contingency, kstwobign  # type: ignore
from scipy.stats import t as t_dist  # type: ignore

from ..dataset import ABCRole, DatasetAdapter, ExperimentData, SmallDataset
from ..dataset.backends import SparkDataset
from ..dataset.roles import StatisticRole, TargetRole
from ..utils import BackendsEnum, NoColumnsError, timeit
from ..utils.constants import CATEGORICAL_TYPES_LIST, NUMBER_TYPES_LIST
from ..utils.errors import NotSuitableFieldError
from ..utils.registry import backend_factory
from .abstract import StatsHypothesisTesting
from .comparators import Chi2Test, TTest, ZTest


@backend_factory.register(TTest, SparkDataset)
class StatsTTest(StatsHypothesisTesting):
    """Two-sample t-test with automatic variance homogeneity check.

    Dynamically selects between Student's t-test (equal variances) and
    Welch's t-test (unequal variances) based on the ratio of standard
    deviations. Uses the base ``StatsComparator.execute()`` pipeline,
    which already handles per-column result storage emulation.
    """
    REQUIRED_STATS: ClassVar[list[str]] = ["mean", "std", "count"]

    def __init__(
            self, 
            grouping_role: ABCRole | None = None,
            compare_by: Literal["groups", "matched_pairs"] = "groups",
            target_roles: ABCRole | None = None,
            baseline_role: ABCRole | None = None,
            reliability: float = 0.05,
            key: Any = "", 
    ):
        """
        Initialize StatsTTest with roles and significance level.

        Args:
            grouping_role: Role defining the grouping column.
            target_roles: Role(s) defining the target numeric column(s).
            reliability: Significance level (alpha) for the test.
            key: Optional identifier for the test instance.
        """
        super().__init__(
            stats=self.REQUIRED_STATS, 
            compare_by=compare_by,
            grouping_role=grouping_role,
            target_roles=target_roles,
            baseline_role=baseline_role,
            key=key, 
            reliability=reliability
        )

    @property
    def search_types(self) -> list[type] | None:
        """
        Return allowed data types for this statistical test.

        Returns:
            List of numeric types supported by the t-test.
        """
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls,
        baseline_stats: dict[str, Any],
        compared_stats: dict[str, Any],
        reliability: float = 0.05,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute t-test logic based on variance similarity heuristic.

        Checks if variances are similar (ratio between 0.5 and 2.0). 
        If similar, uses pooled variance (Student's); otherwise uses Welch's approximation.
        Returns None values if variances are zero.

        Args:
            baseline_stats: {"mean": float, "std": float, "count": int} for baseline group.
            compared_stats: {"mean": float, "std": float, "count": int} for compared group.
            reliability: Significance threshold for the pass flag.

        Returns:
            {"p-value": float | None, "statistic": float | None, "pass": bool | None}
        """
        n1 = baseline_stats["count"]
        n2 = compared_stats["count"]

        # Edge case: insufficient sample size
        if n1 < 2 or n2 < 2:
            return {"p-value": None, "statistic": None, "pass": None}

        current_variances = (baseline_stats["std"]**2, compared_stats["std"]**2)
        current_means = (baseline_stats["mean"], compared_stats["mean"])
        current_sizes = (n1, n2)

        # Edge case: both variances are zero
        if current_variances[0] == 0 and current_variances[1] == 0:
            # If variances are zero, check equality of means
            if current_means[0] == current_means[1]:
                return {"p-value": 1.0, "statistic": 0.0, "pass": True}
            else:
                return {"p-value": 0.0, "statistic": float("inf"), "pass": False}

        # Edge case: one of the variances is zero
        if current_variances[0] == 0 or current_variances[1] == 0:
            # Use Welch's t-test with one zero variance
            similar_var = False
        else:
            similar_var = (current_variances[0] < 2 * current_variances[1] and
                          current_variances[0] > 0.5 * current_variances[1])

        t_stat = cls._t_statistics(
                        n_list=current_sizes,
                        s_list=current_variances,
                        mean_list=current_means,
                        similar_var=similar_var
                    )

        de_fr = cls._degrees_of_freedom(
                    n_list=current_sizes,
                    s_list=current_variances,
                    similar_var=similar_var
        )

        p_value = float(2 * t_dist.sf(abs(t_stat), de_fr))

        return {"p-value": p_value,
                "statistic": float(t_stat),
                "pass": p_value < reliability,}

    @staticmethod
    def _t_statistics(n_list: tuple, 
                      s_list: tuple, 
                      mean_list: tuple, 
                      similar_var: bool = True) -> float:
        """
        Calculate t-statistic based on variance assumption.

        Args:
            n_list: Tuple of sample sizes (n1, n2).
            s_list: Tuple of standard deviations (std1, std2).
            mean_list: Tuple of means (mean1, mean2).
            similar_var: If True, use pooled variance (Student's). If False, use Welch's SE.

        Returns:
            Calculated t-statistic value.
        """
        if similar_var:
            sp = sqrt(
                (
                    (n_list[0] - 1) * s_list[0] + 
                    (n_list[1] - 1) * s_list[1] 
                ) / ( n_list[0] + n_list[1] - 2)
            )
            t_stat = (mean_list[0] - mean_list[1]) / (sp * sqrt(1 / n_list[0] + 1 / n_list[1]))
        else:
            s_delta =sqrt(s_list[0] / n_list[0] + s_list[1] / n_list[1])
            t_stat = (mean_list[0] - mean_list[1]) / s_delta

        return t_stat

    @staticmethod
    def _degrees_of_freedom(n_list: tuple, 
                       s_list: tuple = (0, 0), 
                       similar_var: bool = True) -> float:
        """
        Calculate degrees of freedom for the t-distribution.

        Args:
            n_list: Tuple of sample sizes (n1, n2).
            s_list: Tuple of standard deviations (used only if similar_var is False).
            similar_var: If True, return n1 + n2 - 2. If False, return Welch-Satterthwaite DF.

        Returns:
            Degrees of freedom value.
        """
        if similar_var:
            return n_list[0] + n_list[1] - 2
        else:
            num = ((s_list[0] / n_list[0] + s_list[1] / n_list[1]) ** 2)
            den = ((s_list[0] / n_list[0]) ** 2 / (n_list[0] - 1) + 
                   (s_list[1] / n_list[1]) ** 2 / (n_list[1] - 1))
            return num / den

@backend_factory.register(Chi2Test, SparkDataset)
class StatsChi2Test(StatsHypothesisTesting):
    """
    Chi-squared test of independence operating on aggregated value counts.

    Computes value_counts per group for each target column via a single
    _compute_stats call per group, then applies Pearson's chi-squared test
    analytically — without requiring access to raw data.

    Builds a contingency table from the value counts of the baseline and compared
    groups, then tests whether the distributions of categorical values are independent
    of group membership.

    Output shape is identical to GroupTTest: rows indexed by compared group name, columns
    are p-value, statistic, pass — making StatsChi2Test a drop-in replacement for GroupTTest
    in pipelines where raw data transfer is expensive (e.g. Spark backend).
    """
    REQUIRED_STATS: ClassVar[list[str]] = ["value_counts"]

    def __init__(
            self,
            grouping_role: ABCRole | None = None,
            compare_by: Literal["groups", "matched_pairs"] = "groups",
            target_roles: ABCRole | None = None,
            baseline_role: ABCRole | None = None,
            reliability: float = 0.05,
            key: Any = "",
    ):
        """
        Initialize StatsChi2Test with roles and reliability.

        Args:
            grouping_role: Role defining the grouping column.
            target_roles: Role(s) defining the target categorical column(s).
            reliability: Significance level (alpha) for the test.
            key: Optional identifier for the test instance.
        """
        super().__init__(
            stats=self.REQUIRED_STATS,
            compare_by=compare_by,
            grouping_role=grouping_role,
            target_roles=target_roles,
            baseline_role=baseline_role,
            key=key,
            reliability=reliability
        )
        self.reliability = reliability

    @property
    def search_types(self) -> list[type] | None:
        """
        Return allowed data types for this statistical test.

        Returns:
            List of categorical types supported by the chi-squared test.
        """
        return CATEGORICAL_TYPES_LIST

    @classmethod
    def _inner_function(
        cls,
        baseline_stats: dict[str, Any],
        compared_stats: dict[str, Any],
        reliability: float = 0.05,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Apply Pearson's chi-squared test to pre-aggregated value counts.

        Args:
            baseline_stats: {"value_counts": dict} for baseline group.
            compared_stats: {"value_counts": dict} for compared group.
            reliability: Significance threshold for the pass flag.

        Returns:
            {"p-value": float | None, "statistic": float | None, "pass": bool | None}
        """
        control_freqs = baseline_stats["value_counts"]
        test_freqs = compared_stats["value_counts"]

        # Edge case: empty value_counts
        if not control_freqs and not test_freqs:
            return {"p-value": None, "statistic": None, "pass": None}

        # Edge case: one of the groups is empty
        if not control_freqs or not test_freqs:
            return {"p-value": None, "statistic": None, "pass": None}

        full_key_set = set(control_freqs.keys()).union(set(test_freqs.keys()))

        # Edge case: only one category across all groups
        if len(full_key_set) < 2:
            return {"p-value": 1.0, "statistic": 0.0, "pass": True}

        contingency_table = np.zeros((2, len(full_key_set)))
        for idx, key in enumerate(full_key_set):
            if key in control_freqs:
                contingency_table[0][idx] = control_freqs[key]
            if key in test_freqs:
                contingency_table[1][idx] = test_freqs[key]

        # Edge case: one of the rows in contingency table is empty (sum = 0)
        if contingency_table[0].sum() == 0 or contingency_table[1].sum() == 0:
            return {"p-value": None, "statistic": None, "pass": None}

        # Edge case: column with zero sum (all zeros in the column)
        col_sums = contingency_table.sum(axis=0)
        if np.any(col_sums == 0):
            non_zero_cols = col_sums > 0
            contingency_table = contingency_table[:, non_zero_cols]
            if contingency_table.shape[1] < 2:
                return {"p-value": 1.0, "statistic": 0.0, "pass": True}

        try:
            statistics = chi2_contingency(contingency_table, **kwargs)
            result = {
                    "p-value": float(statistics[1]),
                    "statistic": float(statistics[0]),
                    "pass": statistics[1] < reliability,
                }
        except ValueError:
            # For example, when all values in the table are identical
            return {"p-value": 1.0, "statistic": 0.0, "pass": True}

        return result

@backend_factory.register(ZTest, SparkDataset)
class StatsZTest(StatsHypothesisTesting):
    """
    Z-test for proportions (approximation of Chi-square for 2x2 table).
    Compares conversion rates between baseline and compared groups.

    For continuous metrics (revenue, spends) use AggTTest instead.
    """
    REQUIRED_STATS: ClassVar[list[str]] = ["count", "sum"]

    def __init__(
            self,
            grouping_role: ABCRole | None = None,
            compare_by: Literal["groups", "matched_pairs"] = "groups",
            target_roles: ABCRole | None = None,
            baseline_role: ABCRole | None = None,
            reliability: float = 0.05,
            key: Any = "",
    ):
        """
        Initialize Chi2Test (Z-test for proportions) with roles and reliability.

        Args:
            grouping_role: Role defining the grouping column.
            target_roles: Role(s) defining the target binary column(s).
            reliability: Significance level (alpha) for the test.
            key: Optional identifier for the test instance.
        """
        super().__init__(
            stats=self.REQUIRED_STATS, 
            compare_by=compare_by,
            grouping_role=grouping_role,
            target_roles=target_roles,
            baseline_role=baseline_role,
            key=key, 
            reliability=reliability
        )

    @property
    def search_types(self) -> list[type] | None:
        """
        Return allowed data types for this statistical test.

        Returns:
            List of numeric types supported (typically integers/floats for binary sums).
        """
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(cls,
                        baseline_stats: dict[str, Any],
                        compared_stats: dict[str, Any],
                        reliability: float = 0.05,
                        **kwargs) -> dict[str, Any]:
        """
        Perform Z-test for two proportions using aggregated counts and sums.

        Calculates pooled proportion, standard error, and z-statistic.
        Handles edge cases: zero counts, probabilities outside [0, 1], zero SE.

        Args:
            baseline_stats: Dict with "count" (total) and "sum" (successes) for baseline.
            compared_stats: Dict with "count" (total) and "sum" (successes) for compared.
            reliability: Significance threshold for the 'pass' flag.
            **kwargs: Additional arguments.

        Returns:
            Dictionary with "p-value", "statistic" (z-score), and "pass".
        """
        n1 = baseline_stats["count"]
        n2 = compared_stats["count"]
        s1 = baseline_stats.get("sum", 0)
        s2 = compared_stats.get("sum", 0)

        if n1 == 0 or n2 == 0:
            return {"p-value": None, "statistic": None, "pass": None}

        p1 = s1 / n1
        p2 = s2 / n2
        p_pool = (s1 + s2) / (n1 + n2)


        if p_pool < 0 or p_pool > 1:
            return {"p-value": None, "statistic": None, "pass": None}

        if p_pool == 0 or p_pool == 1:
            return {"p-value": None, "statistic": None, "pass": None}

        variance = p_pool * (1 - p_pool) * (1/n1 + 1/n2)
        se = math.sqrt(max(0, variance))

        if se == 0:
            return {"p-value": None, "statistic": None, "pass": None}

        z_stat = (p1 - p2) / se
        p_value = float(2 * t_dist.sf(abs(z_stat), n1 + n2 - 2))

        return {
            "p-value": [p_value],
            "statistic": [float(z_stat)],
            "pass": [p_value < reliability],
        }


class StatsKSTest(StatsHypothesisTesting):
    """Kolmogorov-Smirnov test on aggregated histograms.

    For Spark: computes per-group histograms, then emulates per-column
    result storage (with the correct ``self.key``) so reporters can
    parse them.

    For Pandas: delegates to ``GroupKSTest`` (``scipy.stats.ks_2samp``).
    """

    REQUIRED_STATS: ClassVar[list[str]] = ["histogram", "count"]

    def __init__(
        self,
        n_bins: int = 2000,
        grouping_role: ABCRole | None = None,
        target_roles: ABCRole | None = None,
        reliability: float = 0.05,
        key: Any = "",
    ):
        """Initializes the stats-based Kolmogorov-Smirnov test.

        Args:
            n_bins: Number of histogram bins to use for the Spark path.
            grouping_role: Role used to identify the grouping column.
            target_roles: Role(s) used to identify target columns.
            reliability: Significance level (alpha) for the test.
            key: Optional identifier key for storing results.
        """
        super().__init__(
            stats=self.REQUIRED_STATS,
            compare_by="groups",
            grouping_role=grouping_role,
            target_roles=target_roles,
            key=key,
            reliability=reliability,
        )
        self.n_bins = n_bins

    @property
    def search_types(self) -> list[type] | None:
        """Returns the expected data types for target columns.

        Returns:
            A list containing ``int`` and ``float``, since the KS test
            operates on numeric data only.
        """
        return NUMBER_TYPES_LIST

    @classmethod
    def _compute_stats(cls, grouped, target_columns, stats=None, **kwargs):
        """Raises ``NotImplementedError``.

        All aggregation for ``StatsKSTest`` happens inside ``execute()``.
        Do not call this method directly.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            "StatsKSTest uses custom execute() logic. "
            "Do not call _compute_stats directly."
        )

    def execute(self, data) -> "ExperimentData":
        """Main entry point. Routes to Spark-optimized or Pandas-fallback path.

        For Spark, delegates to ``_execute_spark``. For Pandas, creates a
        ``GroupKSTest`` delegate with the same ID so pipeline lookups
        remain consistent.

        Args:
            data: The ``ExperimentData`` container.

        Returns:
            The updated ``ExperimentData`` with test results stored in
            ``analysis_tables``.

        Raises:
            NoColumnsError: If no target columns are found.
            NotSuitableFieldError: If the grouping field is not suitable.
        """
        fields = self._get_fields_data(data)
        group_field_data = fields["group_field"]
        target_fields_data = fields["target_fields"]

        if len(target_fields_data.columns) == 0:
            if data.ds.tmp_roles:
                return data
            raise NoColumnsError(TargetRole().role_name)

        if len(group_field_data.columns) != 1:
            raise NotSuitableFieldError(group_field_data, "Grouping")

        self.key = str(
            target_fields_data.columns[0]
            if len(target_fields_data.columns) == 1
            else list(target_fields_data.columns)
        )

        if data.ds.backend_type == BackendsEnum.spark:
            return self._execute_spark(
                data,
                group_col=group_field_data.columns[0],
                target_cols=list(target_fields_data.columns),
            )
        else:
            # Pandas fallback: scipy ks_2samp is faster for small data
            from .hypothesis_testing import GroupKSTest

            delegate = GroupKSTest(
                compare_by="groups",
                grouping_role=self.grouping_role,
                target_role=self.target_roles,
                reliability=self.reliability,
                key=self.key,
            )
            delegate._id = self._id  # Preserve ID for pipeline lookups
            return delegate.execute(data)

    @timeit(level="SPARK", prefix="KS_SPARK")
    def _execute_spark(self, data, group_col: str, target_cols: list[str]):
        """Executes the Kolmogorov-Smirnov test using the Spark-optimized path.

        Delegates histogram aggregation to ``StatsKSTestExtension``, which
        computes per-group histograms for all target columns in a fixed
        number of Spark jobs (global bounds → counts → bucket histograms).
        The KS statistic and p-value are then calculated from the
        pre-aggregated histograms via ``_inner_function``.

        The method performs the following steps:

        1. Computes per-group histograms and observation counts for every
        target column via ``StatsKSTestExtension.calc()``.
        2. If fewer than two groups are present, stores empty results for
        each target column and returns early.
        3. For each target column, runs ``_inner_function`` pairwise
        (baseline vs. each compared group) and appends the results into
        a single ``SmallDataset`` whose index is set to the compared
        group names.
        4. Stores the per-column result under ``self.key`` in
        ``analysis_tables``.

        Note:
            ``self.key`` is temporarily set to the current column name
            during the per-column loop so that each result is stored under
            the correct identifier.  After the loop, ``self.key`` is
            restored to the full list of target columns (or the single
            column name).

        Args:
            data: The ``ExperimentData`` container holding the datasets.
            group_col: Name of the column that defines group membership.
            target_cols: List of numeric target column names to test.

        Returns:
            The updated ``ExperimentData`` with KS test results stored in
            ``analysis_tables`` under per-column keys.
        """
        from ..extensions.stats_hypothesis_testing import StatsKSTestExtension

        subset = data.ds[[group_col] + target_cols]

        ext = StatsKSTestExtension(n_bins=self.n_bins, reliability=self.reliability)
        all_group_stats = ext.calc(
            data=subset,
            group_col=group_col,
            target_cols=target_cols,
        )

        group_names = sorted(all_group_stats.keys(), key=str)
        if len(group_names) < 2:
            for col in target_cols:
                self.key = str(col)
                self._set_value(data, SmallDataset.create_empty())
            return data

        baseline_name = group_names[0]
        for col in target_cols:
            self.key = str(col)
            col_results = []
            for compared_name in group_names[1:]:
                b_stats = all_group_stats.get(baseline_name, {}).get(
                    col, {"histogram": {}, "count": 0}
                )
                c_stats = all_group_stats.get(compared_name, {}).get(
                    col, {"histogram": {}, "count": 0}
                )
                result = self._inner_function(
                    b_stats, c_stats, reliability=self.reliability
                )
                col_results.append(
                    DatasetAdapter.to_dataset(result, StatisticRole())
                )
            if col_results:
                result_dataset = col_results[0].append(col_results[1:])
                result_dataset.index = [str(g) for g in group_names[1:]]
                self._set_value(data, result_dataset)

        self.key = str(target_cols if len(target_cols) > 1 else target_cols[0])
        return data

    @classmethod
    def _inner_function(
        cls,
        baseline_stats: dict[str, Any],
        compared_stats: dict[str, Any],
        reliability: float = 0.05,
        **kwargs,
    ) -> dict[str, Any]:
        """Computes the KS test from pre-aggregated histograms.

        Iterates through the union of histogram buckets, computes the
        empirical CDFs for both groups, and finds the maximum absolute
        difference (D-statistic). The p-value is calculated using the
        asymptotic Kolmogorov distribution (``kstwobign``) with
        Stephens' correction for finite samples.

        Args:
            baseline_stats: Dictionary with ``histogram`` (bucket -> count)
                and ``count`` for the baseline group.
            compared_stats: Dictionary with ``histogram`` and ``count``
                for the compared group.
            reliability: Significance level (alpha) used to compute the
                ``pass`` flag.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            A dictionary with keys ``p-value``, ``statistic``, and ``pass``.
            Returns ``None`` values when either group has zero observations,
            and ``(1.0, 0.0, True)`` when the distributions are identical.
        """
        hist1 = baseline_stats.get("histogram", {})
        hist2 = compared_stats.get("histogram", {})
        n1 = baseline_stats.get("count", 0)
        n2 = compared_stats.get("count", 0)

        if n1 == 0 or n2 == 0:
            return {"p-value": None, "statistic": None, "pass": None}

        all_buckets = sorted(set(hist1.keys()) | set(hist2.keys()))

        if len(all_buckets) == 0:
            return {"p-value": 1.0, "statistic": 0.0, "pass": True}

        cum1 = 0
        cum2 = 0
        d_stat = 0.0
        for bucket in all_buckets:
            cum1 += hist1.get(bucket, 0)
            cum2 += hist2.get(bucket, 0)
            diff = abs(cum1 / n1 - cum2 / n2)
            if diff > d_stat:
                d_stat = diff
        d_stat = float(d_stat)

        if d_stat == 0.0:
            return {"p-value": 1.0, "statistic": 0.0, "pass": True}

        try:
            en = np.sqrt(n1 * n2 / (n1 + n2))
            p_value = float(kstwobign.sf((en + 0.12 + 0.11 / en) * d_stat))
        except Exception:
            p_value = 0.0

        return {
            "p-value": p_value,
            "statistic": d_stat,
            "pass": p_value < reliability,
        }
