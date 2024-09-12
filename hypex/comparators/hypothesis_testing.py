from typing import Optional, List

from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu  # type: ignore

from .abstract import StatHypothesisTesting
from ..dataset import Dataset
from ..extensions.scipy_stats import (
    TTestExtensionExtension,
    KSTestExtensionExtension,
    UTestExtensionExtension,
    Chi2TestExtensionExtension,
)
from ..utils.constants import NUMBER_TYPES_LIST


class TTest(StatHypothesisTesting):
    @property
    def search_types(self) -> Optional[List[type]]:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Dataset:
        return TTestExtensionExtension(kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )


class KSTest(StatHypothesisTesting):

    @property
    def search_types(self) -> Optional[List[type]]:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Dataset:
        return KSTestExtensionExtension(kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )


class UTest(StatHypothesisTesting):

    @property
    def search_types(self) -> Optional[List[type]]:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Dataset:
        return UTestExtensionExtension(kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )


class Chi2Test(StatHypothesisTesting):

    @property
    def search_types(self) -> Optional[List[type]]:
        return [str]

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Dataset:
        return Chi2TestExtensionExtension(
            reliability=kwargs.get("reliability", 0.05)
        ).calc(data, other=test_data, **kwargs)
