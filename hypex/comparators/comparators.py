from typing import Dict, Optional, Union, List

from hypex.comparators.abstract import GroupComparator
from hypex.dataset import Dataset, ABCRole
from hypex.utils import SpaceEnum
from hypex.utils.adapter import Adapter
from hypex.utils.constants import NUMBER_TYPES_LIST


class GroupDifference(GroupComparator):
    def __init__(
        self,
        grouping_role: ABCRole = None,
        space: SpaceEnum = SpaceEnum.auto,
        key: str = "",
    ):
        super().__init__(grouping_role=grouping_role, space=space, key=key)

    @property
    def search_types(self) -> Optional[List[type]]:
        return NUMBER_TYPES_LIST

    @staticmethod
    def _to_dataset(data: Dict, **kwargs) -> Dataset:
        return Adapter.dict_to_dataset(data=data)

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Optional[Dataset] = None,
        **kwargs,
    ) -> Dict:
        # test_data = cls._to_dataset(test_data=test_data)
        control_mean = data.mean()
        test_mean = test_data.mean()

        return {
            "control mean": control_mean,
            "test mean": test_mean,
            "difference": test_mean - control_mean,
            "difference %": (test_mean / control_mean - 1) * 100,
        }


class GroupSizes(GroupComparator):
    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Dict:
        size_a = len(data)
        size_b = len(test_data) if isinstance(test_data, Dataset) else 0

        return {
            "control size": size_a,
            "test size": size_b,
            "control size %": (size_a / (size_a + size_b)) * 100,
            "test size %": (size_b / (size_a + size_b)) * 100,
        }

    @staticmethod
    def _to_dataset(data: Dict, **kwargs) -> Dataset:
        return Adapter.dict_to_dataset(data=data)


class ATE(GroupComparator):
    def __init__(
        self, grouping_role: ABCRole, space: SpaceEnum = SpaceEnum.auto, key: str = ""
    ):
        super().__init__(
            grouping_role=grouping_role,
            space=space,
            key=key,
        )

    @property
    def search_types(self) -> Optional[List[type]]:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> float:
        size_a = len(data)
        size_b = len(test_data)
        control_mean = data.mean()
        test_mean = test_data.mean()

        ate = (size_a / (size_a + size_b)) * control_mean + (
            size_b / (size_a + size_b)
        ) * test_mean

        return ate.iloc[0]

    @staticmethod
    def _to_dataset(data: float, **kwargs) -> Dataset:
        return Adapter.float_to_dataset(name="ATE", data=data)
