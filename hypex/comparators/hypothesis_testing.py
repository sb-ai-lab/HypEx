from __future__ import annotations

from ..dataset import Dataset
from ..extensions.scipy_stats import (
    Chi2TestExtension,
    KSTestExtension,
    TTestExtension,
    UTestExtension,
)
from ..utils.constants import NUMBER_TYPES_LIST
from .abstract import StatHypothesisTesting


class TTest(StatHypothesisTesting):
    """Two-sample t-test for numeric targets.

    Compares group means using Welch's t-test (unequal variances assumed).
    Operates on raw data slices via scipy. For Spark workloads prefer the
    aggregated-stats variant exposed through :class:`hypex.comparators.AggTTest`.

    Args:
        compare_by: Comparison mode (``"groups"``, ``"columns"``, etc.).
        grouping_role: Role that identifies the group column.
        target_role: Role that identifies the numeric target column(s).
        reliability: Significance level α (default 0.05).
        key: Optional label for this test instance.
    """

    @property
    def search_types(self) -> list[type] | None:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> Dataset:
        return TTestExtension(kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )


class KSTest(StatHypothesisTesting):
    """Two-sample Kolmogorov-Smirnov test for numeric targets.

    Tests whether two groups are drawn from the same distribution without
    assuming normality. Operates on raw data slices via scipy.

    Args:
        compare_by: Comparison mode (``"groups"``, ``"columns"``, etc.).
        grouping_role: Role that identifies the group column.
        target_role: Role that identifies the numeric target column(s).
        reliability: Significance level α (default 0.05).
        key: Optional label for this test instance.
    """

    @property
    def search_types(self) -> list[type] | None:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> Dataset:
        return KSTestExtension(kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )


class UTest(StatHypothesisTesting):
    """Mann-Whitney U test (Wilcoxon rank-sum) for numeric targets.

    Non-parametric alternative to the t-test; compares rank distributions
    rather than means. Operates on raw data slices via scipy.

    Args:
        compare_by: Comparison mode (``"groups"``, ``"columns"``, etc.).
        grouping_role: Role that identifies the group column.
        target_role: Role that identifies the numeric target column(s).
        reliability: Significance level α (default 0.05).
        key: Optional label for this test instance.
    """

    @property
    def search_types(self) -> list[type] | None:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> Dataset:
        return UTestExtension(kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )


class Chi2Test(StatHypothesisTesting):
    """Chi-square test of independence for categorical targets.

    Tests whether the distribution of a categorical column differs
    significantly between groups. Operates on raw data slices via scipy.

    Args:
        compare_by: Comparison mode (``"groups"``, ``"columns"``, etc.).
        grouping_role: Role that identifies the group column.
        target_role: Role that identifies the categorical target column(s).
        reliability: Significance level α (default 0.05).
        key: Optional label for this test instance.
    """

    @property
    def search_types(self) -> list[type] | None:
        return [str]

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> Dataset:
        return Chi2TestExtension(reliability=kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )
