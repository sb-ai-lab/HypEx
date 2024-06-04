from typing import Dict, Any, Optional

from hypex.comparators.abstract import GroupComparator
from hypex.dataset import TempTargetRole, Dataset
from hypex.utils import FieldKeyTypes
from hypex.utils.adapter import Adapter


class GroupDifference(GroupComparator):
    @staticmethod
    def _inner_function(data: Dataset, test_data: Dataset, target_field: Optional[FieldKeyTypes], **kwargs) -> Dict:
        control_mean = data.mean()
        test_mean = test_data.mean()

        return {
            f"{target_field} control mean": control_mean,
            f"{target_field} test mean": test_mean,
            f"{target_field} difference": test_mean - control_mean,
            f"{target_field} difference %": (test_mean / control_mean - 1) * 100,
        }

    @staticmethod
    def _to_dataset(data: Dict, **kwargs) -> Dataset:
        return Adapter.dict_to_dataset(data=data)


class GroupSizes(GroupComparator):
    @staticmethod
    def _inner_function(data: Dataset, test_data: Dataset, **kwargs) -> Dict:
        size_a = len(data)
        size_b = len(test_data)

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

    @staticmethod
    def _inner_function(data: Dataset, test_data: Dataset, **kwargs) -> float:
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
