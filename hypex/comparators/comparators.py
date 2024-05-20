from typing import Dict, Any

from hypex.comparators.abstract import GroupComparator
from hypex.dataset import TempTargetRole


class GroupDifference(GroupComparator):
    def _comparison_function(self, control_data, test_data) -> Dict[str, Any]:
        target_field = control_data.get_columns_by_roles(
            TempTargetRole(), tmp_role=True
        )[0]
        control_mean = control_data.mean()
        test_mean = test_data.mean()

        return {
            f"{target_field} control mean": control_mean,
            f"{target_field} test mean": test_mean,
            f"{target_field} difference": test_mean - control_mean,
            f"{target_field} difference %": (test_mean / control_mean - 1) * 100,
        }


class GroupSizes(GroupComparator):
    def _comparison_function(self, control_data, test_data) -> Dict[str, Any]:
        size_a = len(control_data)
        size_b = len(test_data)

        return {
            "control size": size_a,
            "test size": size_b,
            "control size %": (size_a / (size_a + size_b)) * 100,
            "test size %": (size_b / (size_a + size_b)) * 100,
        }


class ATE(GroupComparator):

    def _comparison_function(self, control_data, test_data) -> Dict[str, Any]:
        target_field = control_data.get_columns_by_roles(
            TempTargetRole(), tmp_role=True
        )[0]
        size_a = len(control_data)
        size_b = len(test_data)
        control_mean = control_data.mean()
        test_mean = test_data.mean()

        ate = (size_a / (size_a + size_b)) * control_mean + (
            size_b / (size_a + size_b)
        ) * test_mean

        return {f"{target_field} ATE": ate.iloc[0]}
