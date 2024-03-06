import numpy as np

from hypex.experiment.experiment import Executor
from hypex.dataset.dataset import ExperimentData


class Shuffle(Executor):
    def __init__(self, random_state: int = None, full_name: str = None, index: int = 0):
        super().__init__(full_name, index)
        self.random_state = random_state

    def generate_params_hash(self):
        return f"{self.random_state}"

    @property
    def __is_transformer(self):
        return True

    def execute(self, data: ExperimentData):
        np.random.seed(self.random_state)
        data.data = np.random.shuffle(data.data)
        return data
