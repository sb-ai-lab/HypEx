from typing import Optional, List, Any

from hypex.dataset import ABCRole, TargetRole, Dataset, ExperimentData
from hypex.dataset.roles import MatchingRole
from hypex.executor import GroupCalculator
from hypex.utils import ExperimentDataEnum


class SMD(GroupCalculator):

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        other: Optional[Dataset] = None,
        **kwargs
    ) -> Any:
        return (data.mean() + other.mean()) / data.std()
