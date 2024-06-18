from typing import Dict, Optional

from hypex.comparators.abstract import GroupComparator
from hypex.dataset import Dataset
from hypex.utils import FieldKeyTypes
from hypex.utils.adapter import Adapter


class GroupDifference(GroupComparator):
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
            f"control mean": control_mean,
            f"test mean": test_mean,
            f"difference": test_mean - control_mean,
            f"difference %": (test_mean / control_mean - 1) * 100,
        }


class GroupSizes(GroupComparator):
    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Dict:
        size_a = len(data)
        size_b = len(test_data) if test_data else 0

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
        return Adapter.value_to_dataset(data=data, column_name="ATE")
