from __future__ import annotations

import math
from typing import Any

from scipy.stats import t as t_dist

from ..dataset import ABCRole
from ..utils.constants import NUMBER_TYPES_LIST
from .abstract import StatsHypothesisTesting, StatsComparator

from math import sqrt

class AggTTest(StatsHypothesisTesting):
    """
    Analytical two-sample t-test operating on aggregated sufficient statistics.

    Computes (mean, std, count) per group for each target column via a single
    _compute_stats call per group, then applies Welch's t-test formula analytically
    — without requiring access to raw data.

    Output shape is identical to GroupTTest: rows indexed by compared group name, columns
    are p-value, statistic, pass — making AggTTest a drop-in replacement for GroupTTest
    in pipelines where raw data transfer is expensive (e.g. Spark backend).
    """

    REQUIRED_STATS = ["mean", "std", "count"]

    def __init__(self,
                 grouping_role: ABCRole | None = None,
                 target_roles: ABCRole | None = None,
                 reliability: float = 0.05,
                 key: Any = ""):
        """
        Initialize AggTTest with specific roles and reliability threshold.

        Args:
            grouping_role: Role defining the grouping column for A/B segments.
            target_roles: Role(s) defining the target numeric column(s) to test.
            reliability: Significance level (alpha) for hypothesis testing (default 0.05).
            key: Optional identifier for the test instance.
        """
        super().__init__(
            stats=self.REQUIRED_STATS,
            grouping_role=grouping_role,
            target_roles=target_roles,
            reliability=reliability,
            key=key,
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
    def _inner_function(cls,
                        baseline_stats: dict[str, Any],
                        compared_stats: dict[str, Any],
                        reliability: float = 0.05,
                        **kwargs) -> dict[str, Any]:
        """
        Apply Welch's t-test formula to pre-aggregated group statistics.

        Calculates t-statistic and p-value using means, variances, and counts
        without accessing raw observations. Handles edge cases (small n, zero SE).

        Args:
            baseline_stats: Dictionary containing "mean", "std", "count" for baseline group.
            compared_stats: Dictionary containing "mean", "std", "count" for compared group.
            reliability: Significance threshold for the 'pass' flag.
            **kwargs: Additional arguments (unused).

        Returns:
            Dictionary with keys:
                - "p-value": Calculated two-sided p-value (float or None).
                - "statistic": Calculated t-statistic (float or None).
                - "pass": Boolean indicating if p-value < reliability (bool or None).
        """
        n1 = baseline_stats["count"]
        m1 = baseline_stats["mean"]
        v1 = baseline_stats["std"]**2
        n2 = compared_stats["count"]
        m2 = compared_stats["mean"]
        v2 = compared_stats["std"]**2

        if n1 < 2 or n2 < 2:
            return {"p-value": None, "statistic": None, "pass": None}

        term1 = v1 / n1
        term2 = v2 / n2
        se = math.sqrt(term1 + term2)

        if se == 0:
            return {"p-value": None, "statistic": None, "pass": None}

        t_stat = (m1 - m2) / se

        # Welch-Satterthwaite degrees of freedom
        df = ((term1 + term2) ** 2 / 
              (term1**2 / (n1 - 1) + term2**2 / (n2 - 1)))
        p_value = float(2 * t_dist.sf(abs(t_stat), df))

        return {
            "p-value": [p_value],
            "statistic": [float(t_stat)],
            "pass": [p_value < reliability],
        }

class StatsTTest(StatsHypothesisTesting):
    """
    Two-sample t-test with automatic variance homogeneity check.

    Dynamically selects between Student's t-test (equal variances) and 
    Welch's t-test (unequal variances) based on the ratio of standard deviations.
    Requires pre-calculated sufficient statistics (mean, std, count).
    """
    REQUIRED_STATS = ["mean", "std", "count"]
    
    def __init__(self, 
                 grouping_role: ABCRole | None = None, 
                 target_roles: ABCRole | None = None, 
                 reliability: float = 0.05,
                 key: Any = ""):
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
    def _inner_function(cls,
                        baseline_stats: dict[str, Any],
                        compared_stats: dict[str, Any],
                        reliability: float = 0.05,
                        **kwargs) -> dict[str, Any]:
        """
        Execute t-test logic based on variance similarity heuristic.

        Checks if variances are similar (ratio between 0.5 and 2.0). 
        If similar, uses pooled variance (Student's); otherwise uses Welch's approximation.
        Returns None values if variances are zero.

        Args:
            baseline_stats: Stats dict for the baseline group.
            compared_stats: Stats dict for the compared group.
            reliability: Significance threshold for the 'pass' flag.
            **kwargs: Additional arguments.

        Returns:
            Dictionary with "p-value", "statistic", and "pass".
        """
    
        current_variances = (baseline_stats["std"], compared_stats["std"])
        current_means = (baseline_stats["mean"], compared_stats["mean"])
        current_sizes =(baseline_stats["count"], compared_stats["count"])
        if current_variances[0] != 0 and current_variances[1] != 0:
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

            p_value = t_dist.sf(abs(t_stat), de_fr) * 2

            return {
                    "p-value": [p_value],
                    "statistic": [float(t_stat)],
                    "pass": [p_value < reliability],
                }
        else:
            return {
                "p-value": [None],
                "statistic": [None],
                "pass": [False],
            }
    
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
            sp = sqrt(((n_list[0] - 1) * s_list[0]**2 + (n_list[1] - 1) * s_list[1]**2) / 
                      (n_list[0] + n_list[1] - 2))
            t_stat = (mean_list[0] - mean_list[1]) / (sp * sqrt(1/n_list[0] + 1/n_list[1]))
        else:
            s_delta = sqrt(s_list[0]**2 / n_list[0] + s_list[1]**2 / n_list[1])
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

            num = (s_list[0]**2 / n_list[0] + s_list[1]**2 / n_list[1])**2
            den = ((s_list[0]**2 / n_list[0])**2 / (n_list[0] - 1) + 
                   (s_list[1]**2 / n_list[1])**2 / (n_list[1] - 1))
            return num / den
        
class StatsChi2Test(StatsHypothesisTesting):
    """
    Z-test for proportions (approximation of Chi-square for 2x2 table).
    Compares conversion rates between baseline and compared groups.
    
    ⚠️ ONLY for BINARY data (0/1, success/failure, conversion).
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