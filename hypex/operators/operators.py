from typing import Optional, Any

from hypex.dataset import Dataset
from hypex.operators.abstract import GroupOperator


class SMD(GroupOperator):
    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Any:
        return (data.mean() + test_data.mean()) / data.std()
