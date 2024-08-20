from typing import Optional

from hypex.dataset import ExperimentData
from hypex.executor.executor import IfExecutor


class IfAAExecutor(IfExecutor):
    def __init__(
        self,
        if_executor,
        else_executor,
        sample_size: Optional[float] = None,
        key: str = "",
    ):
        self.sample_size = sample_size
        super().__init__(if_executor, else_executor, key)

    def execute(self, data: ExperimentData) -> ExperimentData:
        if self.sample_size is not None:
            return data
