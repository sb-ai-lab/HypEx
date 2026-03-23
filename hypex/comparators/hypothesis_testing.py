from __future__ import annotations

from ..dataset import Dataset
from ..extensions.scipy_stats import (
    GroupChi2TestExtension,
    GroupKSTestExtension,
    GroupTTestExtension,
    GroupUTestExtension,
)
from ..utils.constants import NUMBER_TYPES_LIST
from .abstract import GroupHypothesisTesting, StatsComparator
from typing import Any

class GroupTTest(GroupHypothesisTesting):
    @property
    def search_types(self) -> list[type] | None:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> Dataset:
        return GroupTTestExtension(kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )

class StatsTTest(StatsComparator):
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
        ...

class GroupKSTest(GroupHypothesisTesting):
    @property
    def search_types(self) -> list[type] | None:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> Dataset:
        return GroupKSTestExtension(kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )

class StatsKSTest(StatsComparator):
    ...

class GroupUTest(GroupHypothesisTesting):
    @property
    def search_types(self) -> list[type] | None:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> Dataset:
        return GroupUTestExtension(kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )


class GroupChi2Test(GroupHypothesisTesting):
    @property
    def search_types(self) -> list[type] | None:
        return [str]

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> Dataset:
        return GroupChi2TestExtension(reliability=kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )
    
class StatsChi2Test(StatsComparator):
    ...
