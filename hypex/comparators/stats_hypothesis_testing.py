from __future__ import annotations

import math
from typing import Any

from scipy.stats import t as t_dist

from ..dataset import ABCRole
from ..utils.constants import NUMBER_TYPES_LIST
from .abstract import StatsHypothesisTesting


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
