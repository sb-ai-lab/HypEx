from __future__ import annotations

import math
import numpy as np
from typing import Any

from scipy.stats import t as t_dist

from ..dataset import ABCRole
from ..utils.constants import NUMBER_TYPES_LIST, CATEGORICAL_TYPES_LIST
from .abstract import StatsHypothesisTesting

from typing import Any, Union
from math import sqrt
from scipy.stats import (
    t,
    chi2_contingency,
)

class AggTTest(StatsHypothesisTesting):
    """
    Analytical two-sample t-test operating on aggregated sufficient statistics.

    Computes (mean, var, count) per group for each target column via a single
    _compute_stats call per group, then applies Welch's t-test formula analytically
    — without requiring access to raw data.

    Output shape is identical to GroupTTest: rows indexed by compared group name, columns
    are p-value, statistic, pass — making AggTTest a drop-in replacement for GroupTTest
    in pipelines where raw data transfer is expensive (e.g. Spark backend).
    """

    REQUIRED_STATS = ["mean", "var", "count"]

    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        target_roles: ABCRole | None = None,
        reliability: float = 0.05,
        key: Any = "",
    ):
        super().__init__(
            stats=self.REQUIRED_STATS,
            grouping_role=grouping_role,
            target_roles=target_roles,
            reliability=reliability,
            key=key,
        )

    @property
    def search_types(self) -> list[type] | None:
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
        Apply Welch's t-test formula to pre-aggregated group statistics.

        Args:
            baseline_stats: {"mean": float, "var": float, "count": int} for baseline group.
            compared_stats: {"mean": float, "var": float, "count": int} for compared group.
            reliability: Significance threshold for the pass flag.

        Returns:
            {"p-value": float | None, "statistic": float | None, "pass": bool | None}
        """
        n1 = baseline_stats["count"]
        m1 = baseline_stats["mean"]
        v1 = baseline_stats["var"]
        n2 = compared_stats["count"]
        m2 = compared_stats["mean"]
        v2 = compared_stats["var"]

        if n1 < 2 or n2 < 2:
            return {"p-value": None, "statistic": None, "pass": None}

        term1 = v1 / n1
        term2 = v2 / n2
        se = math.sqrt(term1 + term2)

        if se == 0:
            return {"p-value": None, "statistic": None, "pass": None}

        t_stat = (m1 - m2) / se

        # Welch-Satterthwaite degrees of freedom
        df = (term1 + term2) ** 2 / (
            term1**2 / (n1 - 1) + term2**2 / (n2 - 1)
        )
        p_value = float(2 * t_dist.sf(abs(t_stat), df))

        return {
            "p-value": p_value,
            "statistic": float(t_stat),
            "pass": p_value < reliability,
        }

class StatsTTest(StatsHypothesisTesting):
    """
    Two-sample t-test operating on aggregated sufficient statistics.

    Computes (mean, std, count) per group for each target column via a single
    _compute_stats call per group, then applies Student's t-test formula analytically
    — without requiring access to raw data.

    Automatically determines whether to use pooled variance (when variances are similar)
    or Welch's approximation (when variances differ significantly).

    Output shape is identical to GroupTTest: rows indexed by compared group name, columns
    are p-value, statistic, pass — making StatsTTest a drop-in replacement for GroupTTest
    in pipelines where raw data transfer is expensive (e.g. Spark backend).
    """
    REQUERED_STATS = ["mean", "std", "count"]

    def __init__(
            self, 
            grouping_role = None, 
            target_roles = None, 
            reliability: float = 0.05,
            key: Any = "", 
    ):
        super().__init__(
            stats=self.REQUERED_STATS, 
            grouping_role=grouping_role, 
            target_roles=target_roles, 
            key=key, 
            calc_kwargs={"reliability": reliability}
        )
        self.reliability = reliability

    @property
    def search_types(self) -> list[type] | None:
        return NUMBER_TYPES_LIST
    
    @classmethod
    def _inner_function(
        cls,
        baseline_stats: dict[str, Any],
        compared_stats: dict[str, Any],
        reliability: float = 0.05,
        **kwargs,
    ) -> dict[str, Any]:
    
        current_variances = (baseline_stats["std"], compared_stats["std"])
        current_means = (baseline_stats["mean"], compared_stats["mean"])
        current_sizes =(baseline_stats["count"], compared_stats["count"])
        if current_variances[0] != 0 and current_variances[1] != 0:
            similar_var = (current_variances[0] < 2 * current_variances[1] and current_variances[0] > 0.5 * current_variances[1])
            t_stat = cls._t_statistics(
                            n_list=current_sizes,
                            s_list=current_variances,
                            mean_list=current_means,
                            similar_var=similar_var
                        )

            de_fr = cls._degree_fredom(
                        n_list=current_sizes,
                        s_list=current_variances,
                        similar_var=similar_var
            )

            p_value = t.sf(abs(t_stat), de_fr) * 2

            return {
                    "p-value": p_value,
                    "statistic": abs(t_stat),
                    "pass": p_value < reliability,
                }
        else:
            return {
                "p-value": None,
                "statistic": None,
                "pass": False,
            }
    
    @staticmethod
    def _t_statistics(n_list: tuple, 
                      s_list: tuple, 
                      mean_list: tuple, 
                      similar_var: bool = True) -> float:
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
    def _degree_fredom(n_list: tuple, 
                       s_list: tuple = (0, 0), 
                       similar_var: bool = True) -> Union[int, float]:
        if similar_var:
            return n_list[0] + n_list[1] - 2
        else:
            de_fr = ((s_list[0] / n_list[0] + s_list[1] / n_list[1]) ** 2) / (
                          (s_list[0] / n_list[0]) ** 2 / (n_list[0] - 1) + (s_list[1] / n_list[1]) ** 2 / (n_list[1] - 1)
                          )
            return de_fr
        
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
            grouping_role = None, 
            target_roles = None, 
            reliability: float = 0.05,
            key: Any = "", 
    ):
        super().__init__(
            stats=self.REQUIRED_STATS, 
            grouping_role=grouping_role, 
            target_roles=target_roles, 
            key=key, 
            calc_kwargs={"reliability": reliability}
        )
        self.reliability = reliability

    @property
    def search_types(self) -> list[type] | None:
        return CATEGORICAL_TYPES_LIST
    
    @classmethod
    def _inner_function(
        cls,
        baseline_stats: dict[str, Any],
        compared_stats: dict[str, Any],
        reliability: float = 0.05,
        **kwargs,
    ) -> dict[str, Any]:
        control_freqs = baseline_stats["value_counts"]
        test_freqs = compared_stats["value_counts"]

        full_key_set = set(control_freqs.keys()).union(set(test_freqs.keys()))
        contingency_table = np.zeros((2, len(full_key_set)))
        for idx, (key) in enumerate(full_key_set):
            if key in control_freqs:
                contingency_table[0][idx] = control_freqs[key]
            if key in test_freqs:
                contingency_table[1][idx] = test_freqs[key]
        print(contingency_table)
        
        statistics= chi2_contingency(contingency_table, **kwargs)
        result = {
                "p-value": statistics[1],
                "statistic": statistics[0],
                "pass": statistics[1] < reliability,
            }
            
        return result