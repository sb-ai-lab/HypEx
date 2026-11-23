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


class Shuffle(Calculator):
    """Shuffle rows in the dataset with optional random state.

    This transformer randomly shuffles the rows in the dataset, useful for
    breaking any temporal or sequential patterns and ensuring randomness
    in downstream analyses.

    Args:
        random_state: Random seed for reproducibility. Defaults to None.
        key: Optional executor identifier.

    Example:
        >>> from hypex.transformers import Shuffle
        >>> shuffle = Shuffle(random_state=42)
        >>> result = shuffle.execute(experiment_data)
    """

    def __init__(
        self,
        random_state: int | None = None,
        key: Any = "",
    ):
        super().__init__(key)
        self.random_state = random_state

    @staticmethod
    def _inner_function(data: Dataset, random_state: int | None = None) -> Dataset:
        return data.shuffle(random_state=random_state)

    def generate_params_hash(self):
        return f"{self.random_state}"

    def execute(self, data: ExperimentData) -> ExperimentData:
        result = data.copy(data=self.calc(data=data.ds, random_state=self.random_state))
        return result
