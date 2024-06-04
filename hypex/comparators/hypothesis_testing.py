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
from ..utils.typings import NumberTypes, CategoricalTypes


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
            search_types=NumberTypes,
            reliability=reliability,
        )

    def _inner_function(
        self, data: Dataset, test_data: Dataset = None
    ) -> Dataset:
        return TTestExtension(self.reliability).calc(data, test_data=test_data)


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
            search_types=NumberTypes,
            reliability=reliability,
        )

    def _inner_function(self, data: Dataset, test_data: Dataset = None) -> Dataset:
        return KSTestExtension(self.reliability).calc(data, test_data=test_data)


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
            search_types=NumberTypes,
            reliability=reliability,
        )

    def _inner_function(self, data: Dataset, test_data: Dataset = None) -> Dataset:
        return UTestExtension(self.reliability).calc(data, test_data=test_data)


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
            search_types=CategoricalTypes,
            reliability=reliability,
        )

    def _inner_function(self, data: Dataset, test_data: Dataset = None) -> Dataset:
        return Chi2TestExtension(reliability=self.reliability).calc(
            data, test_data=test_data
        )
