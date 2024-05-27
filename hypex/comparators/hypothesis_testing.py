from typing import Union

from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu  # type: ignore

from .abstract import StatHypothesisTesting
from ..dataset import Dataset, ABCRole
from ..extensions.hypothesis_testing import (
    TTestExtension,
    KSTestExtension,
    UTestExtension,
    Chi2TestExtension,
)
from ..utils import SpaceEnum


class TTest(StatHypothesisTesting):
    def __init__(
        self,
        grouping_role: Union[ABCRole, None] = None,
        space: SpaceEnum = SpaceEnum.auto,
        reliability: float = 0.05,
    ):
        super().__init__(
            grouping_role=grouping_role,
            space=space,
            search_types=[int, float],
            reliability=reliability,
        )

    def _comparison_function(
        self, control_data: Dataset, test_data: Dataset
    ) -> Dataset:
        return TTestExtension(self.reliability).calc(control_data, test_data)


class KSTest(StatHypothesisTesting):
    def __init__(
        self,
        grouping_role: Union[ABCRole, None] = None,
        space: SpaceEnum = SpaceEnum.auto,
        reliability: float = 0.05,
    ):
        super().__init__(
            grouping_role=grouping_role,
            space=space,
            search_types=[int, float],
            reliability=reliability,
        )

    def _comparison_function(self, control_data, test_data) -> Dataset:
        return KSTestExtension(self.reliability).calc(control_data, test_data)


class UTest(StatHypothesisTesting):
    def __init__(
        self,
        grouping_role: Union[ABCRole, None] = None,
        space: SpaceEnum = SpaceEnum.auto,
        reliability: float = 0.05,
    ):
        super().__init__(
            grouping_role=grouping_role,
            space=space,
            search_types=[int, float],
            reliability=reliability,
        )

    def _comparison_function(self, control_data, test_data) -> Dataset:
        return UTestExtension(self.reliability).calc(control_data, test_data)


class Chi2Test(StatHypothesisTesting):
    def __init__(
        self,
        grouping_role: Union[ABCRole, None] = None,
        space: SpaceEnum = SpaceEnum.auto,
        reliability: float = 0.05,
    ):
        super().__init__(
            grouping_role=grouping_role,
            space=space,
            search_types=str,
            reliability=reliability,
        )

    def _comparison_function(self, control_data, test_data) -> Dataset:
        return Chi2TestExtension(reliability=self.reliability).calc(
            control_data, test_data
        )
