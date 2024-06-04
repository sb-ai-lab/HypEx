from typing import Dict, Any, Union

from hypex.comparators.abstract import GroupComparator
from hypex.dataset import TempTargetRole, ABCRole
from hypex.utils import SpaceEnum
from hypex.utils.typings import NumberTypes


class GroupDifference(GroupComparator):
    def __init__(
        self,
        grouping_role: Union[ABCRole, None] = None,
        space: SpaceEnum = SpaceEnum.auto,
    ):
        super().__init__(
            grouping_role=grouping_role, space=space, search_types=NumberTypes
        )

    def _comparison_function(self, control_data, test_data) -> Dict[str, Any]:
        target_field = control_data.search_columns(TempTargetRole(), tmp_role=True)[0]
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
    def __init__(
        self,
        grouping_role: Union[ABCRole, None] = None,
        space: SpaceEnum = SpaceEnum.auto,
    ):
        super().__init__(
            grouping_role=grouping_role, space=space, search_types=NumberTypes
        )

    def _comparison_function(self, control_data, test_data) -> Dict[str, Any]:
        target_field = control_data.search_columns(TempTargetRole(), tmp_role=True)[0]
        size_a = len(control_data)
        size_b = len(test_data)
        control_mean = control_data.mean()
        test_mean = test_data.mean()

        ate = (size_a / (size_a + size_b)) * control_mean + (
            size_b / (size_a + size_b)
        ) * test_mean

        return {f"{target_field} ATE": ate.iloc[0]}
