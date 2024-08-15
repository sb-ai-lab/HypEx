from typing import Dict, Optional, List, Literal, Union

from hypex.comparators.abstract import Comparator
from hypex.dataset import Dataset, ABCRole
from hypex.utils.constants import NUMBER_TYPES_LIST


class GroupDifference(Comparator):
    def __init__(
        self,
        compare_by: Literal[
            "groups", "columns", "columns_in_groups", "cross"
        ] = "groups",
        grouping_role: Optional[ABCRole] = None,
        target_roles: Union[ABCRole, List[ABCRole], None] = None,
    ):
        super().__init__(
            compare_by=compare_by,
            grouping_role=grouping_role,
            target_roles=target_roles,
        )

    @property
    def search_types(self) -> Optional[List[type]]:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Optional[Dataset] = None,
        **kwargs,
    ) -> Dict:
        test_data = cls._check_test_data(test_data)
        control_mean = data.mean()
        test_mean = test_data.mean()

        return {
            "control mean": control_mean,
            "test mean": test_mean,
            "difference": test_mean - control_mean,
            "difference %": (test_mean / control_mean - 1) * 100,
        }


class GroupSizes(Comparator):
    def __init__(
        self,
        compare_by: Literal[
            "groups", "columns", "columns_in_groups", "cross"
        ] = "groups",
        grouping_role: Optional[ABCRole] = None,
    ):
        super().__init__(
            compare_by=compare_by,
            grouping_role=grouping_role,
            target_roles=grouping_role,
        )

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
