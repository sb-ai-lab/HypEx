from typing import Any, Optional

from ..dataset import Dataset, ExperimentData
from ..executor.executor import Calculator


class Shuffle(Calculator):
    def __init__(
        self,
        random_state: Optional[int] = None,
        key: Any = "",
    ):
        super().__init__(key)
        self.random_state = random_state

    @staticmethod
    def _inner_function(data: Dataset, random_state: Optional[int] = None) -> Dataset:
        return data.shuffle(random_state=random_state)

    def generate_params_hash(self):
        return f"{self.random_state}"

    def execute(self, data: ExperimentData) -> ExperimentData:
        result = data.copy(data=self.calc(data=data.ds, random_state=self.random_state))
        return result
