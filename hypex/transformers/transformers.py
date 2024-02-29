from hypex.experiment.experiment import Executor
from hypex.dataset.dataset import ExperimentData


class Shuffle(Executor):
    def __init__(self, random_state: int=0, full_name: str = None, index: int = 0):
        super().__init__(full_name, index)
        self.random_state = random_state

    def execute(self, data:ExperimentData):

        return data