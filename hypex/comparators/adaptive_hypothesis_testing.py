from __future__ import annotations

from ..utils import BackendsEnum
from .abstract import AdaptiveHypothesisTest
from .hypothesis_testing import GroupChi2Test, GroupKSTest, GroupTTest, GroupUTest
from .stats_hypothesis_testing import StatsTTest, StatsChi2Test, StatsKSTest


class TTest(AdaptiveHypothesisTest):
    """
    Backend-adaptive t-test.

    Pandas → GroupTTest (raw-data Welch/Student via scipy).
    Spark  → GroupTTest  (analytical Welch on aggregated stats — single Spark job).
    """

    BACKEND_MAP = {
        BackendsEnum.pandas: GroupTTest,
        BackendsEnum.spark: StatsTTest,
    }


class Chi2Test(AdaptiveHypothesisTest):
    """
    Backend-adaptive proportion (chi-square / z) test.

    Pandas → GroupChi2Test (raw-data chi-square via scipy).
    Spark  → GroupChi2Test (z-test for proportions on aggregated count+sum).
    """

    BACKEND_MAP = {
        BackendsEnum.pandas: GroupChi2Test,
        BackendsEnum.spark: StatsChi2Test,
    }


class KSTest(AdaptiveHypothesisTest):
    """
    Backend-adaptive Kolmogorov-Smirnov test.
    Pandas → GroupKSTest (raw-data scipy ks_2samp).
    Spark  → StatsKSTest (histogram-based, single groupBy per target).
    """

    BACKEND_MAP = {
        BackendsEnum.pandas: GroupKSTest,
        BackendsEnum.spark: StatsKSTest,
    }


class UTest(AdaptiveHypothesisTest):
    """
    Backend-adaptive Mann-Whitney U test.

    Uses GroupUTest for all backends (no aggregated-stats version available).
    """

    BACKEND_MAP = {
        BackendsEnum.pandas: GroupUTest,
        BackendsEnum.spark: GroupUTest,
    }