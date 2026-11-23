from __future__ import annotations

from typing import Any

from ..dataset import Dataset, ExperimentData
from ..executor.executor import Calculator


class Shuffle(Calculator):
    def __init__(
        self,
        random_state: int | None = None,
        key: Any = "",
    ):
        self.random_state = random_state
        super().__init__(key)

    @staticmethod
    def _inner_function(data: Dataset, random_state: int | None = None) -> Dataset:
        return data.sample(frac=1.0, random_state=random_state)

    def _generate_params_hash(self):
        self._params_hash = f"{self.random_state}"

    def execute(self, data: ExperimentData) -> ExperimentData:
        result = data.copy(data=self.calc(data=data.ds, random_state=self.random_state))
        return result
