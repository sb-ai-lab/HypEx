from __future__ import annotations

import math
import numpy as np
from typing import Any

from scipy.stats import t as t_dist, chi2_contingency

from ..dataset import ABCRole
from ..utils.constants import NUMBER_TYPES_LIST, CATEGORICAL_TYPES_LIST
from .abstract import StatsHypothesisTesting

from math import sqrt

class StatsTTest(StatsHypothesisTesting):
    """
    Two-sample t-test with automatic variance homogeneity check.

    Dynamically selects between Student's t-test (equal variances) and 
    Welch's t-test (unequal variances) based on the ratio of standard deviations.
    Requires pre-calculated sufficient statistics (mean, std, count).
    """
    REQUIRED_STATS = ["mean", "std", "count"]

    def __init__(
            self, 
            grouping_role: ABCRole | None = None,
            target_roles: ABCRole | None = None, 
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
            grouping_role=grouping_role, 
            target_roles=target_roles, 
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
    REQUIRED_STATS = ["value_counts"]

    def __init__(
            self,
            grouping_role: ABCRole | None = None,
            target_roles: ABCRole | None = None,
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
            grouping_role=grouping_role,
            target_roles=target_roles,
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
    
class StatsZTest(StatsHypothesisTesting):
    """
    Z-test for proportions (approximation of Chi-square for 2x2 table).
    Compares conversion rates between baseline and compared groups.
    
    For continuous metrics (revenue, spends) use AggTTest instead.
    """
    REQUIRED_STATS = ["count", "sum"]

    def __init__(self, 
                 grouping_role: ABCRole | None = None, 
                 target_roles: ABCRole | None = None, 
                 reliability: float = 0.05,
                 key: Any = ""):
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
            grouping_role=grouping_role, 
            target_roles=target_roles, 
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
