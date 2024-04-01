from typing import Any, Union

from hypex.dataset.dataset import ExperimentData
from hypex.experiment.experiment import Executor


class Shuffle(Executor):
    def __init__(
        self,
        random_state: Union[int, None] = None,
        full_name: Union[None, str] = None,
        key: Any = 0,
    ):
        super().__init__(full_name, key)
        self.random_state = random_state

    def generate_params_hash(self):
        return f"{self.random_state}"

    @property
    def __is_transformer(self):
        return True

    def execute(self, data: ExperimentData) -> ExperimentData:
        data.data = data.data.sample(frac=1, random_state=self.random_state)
        return data
