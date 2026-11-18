from __future__ import annotations

from typing import Any, cast

from ..dataset import Dataset, ExperimentData
from ..executor.executor import Calculator


class Shuffle(Calculator):
    def __init__(
        self,
        random_state: int | None = None,
        key: Any = "",
    ):
        super().__init__(key)
        self.random_state = random_state

    @staticmethod
    def _inner_function(data: Dataset, random_state: int | None = None) -> Dataset:
        # Preserve the legacy optional shuffle API without adding an implementation.
        return data.shuffle(random_state=random_state)  # type: ignore[attr-defined]

    def generate_params_hash(self):
        return f"{self.random_state}"

    def execute(self, data: ExperimentData) -> ExperimentData:
        result = data.copy(
            data=self.calc(data=cast(Dataset, data.ds), random_state=self.random_state)
        )
        return result
