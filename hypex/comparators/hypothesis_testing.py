from typing import Union, Optional, List

from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu  # type: ignore

from .abstract import StatHypothesisTesting
from ..dataset import Dataset, ABCRole
from ..extensions.scipy_stats import (
    TTestExtension,
    KSTestExtension,
    UTestExtension,
    Chi2TestExtension,
)
from ..utils import SpaceEnum
from ..utils.constants import NUMBER_TYPES_LIST


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
            reliability=reliability,
        )

    @property
    def search_types(self) -> Optional[List[type]]:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Dataset:
        return TTestExtension(kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )


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
            reliability=reliability,
        )

    @property
    def search_types(self) -> Optional[List[type]]:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Dataset:
        return KSTestExtension(kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )


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
            reliability=reliability,
        )

    @property
    def search_types(self) -> Optional[List[type]]:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Dataset:
        return UTestExtension(kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )


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
            reliability=reliability,
        )

    @property
    def search_types(self) -> Optional[List[type]]:
        return [str]

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Dataset:
        return Chi2TestExtension(reliability=kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )
