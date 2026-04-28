from __future__ import annotations

from ..dataset import Dataset
from ..dataset.backends import PandasDataset, SparkDataset
from ..extensions.scipy_stats import (
    GroupChi2TestExtension,
    GroupKSTestExtension,
    GroupTTestExtension,
    GroupUTestExtension,
)
from ..utils.constants import NUMBER_TYPES_LIST
from ..utils.registry import backend_factory
from .abstract import GroupHypothesisTesting, StatsComparator
from .comparators import TTest, KSTest, UTest, Chi2Test

@backend_factory.register(TTest, PandasDataset)
class GroupTTest(GroupHypothesisTesting):
    @property
    def search_types(self) -> list[type] | None:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> Dataset:
        test_cls = backend_factory.resolve_backend(GroupTTestExtension, data)
        return test_cls(kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )

@backend_factory.register(KSTest, [PandasDataset, SparkDataset])
class GroupKSTest(GroupHypothesisTesting):
    @property
    def search_types(self) -> list[type] | None:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> Dataset:
        test_cls = backend_factory.resolve_backend(GroupKSTestExtension, data)
        return test_cls(kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )

@backend_factory.register(UTest, [PandasDataset, SparkDataset])
class GroupUTest(GroupHypothesisTesting):
    @property
    def search_types(self) -> list[type] | None:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> Dataset:
        test_cls = backend_factory.resolve_backend(GroupUTestExtension, data)
        return test_cls(kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )

@backend_factory.register(Chi2Test, PandasDataset)
class GroupChi2Test(GroupHypothesisTesting):
    @property
    def search_types(self) -> list[type] | None:
        return [str]

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> Dataset:
        test_cls = backend_factory.resolve_backend(GroupChi2TestExtension, data)
        return test_cls(reliability=kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )