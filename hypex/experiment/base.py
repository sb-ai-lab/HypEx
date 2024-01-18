from typing import Iterable

from hypex.pipelines.pipeline import Pipeline, Executor

class Experiment(Pipeline):
    pass

class ExperimentMulti(Executor):

    def __init__(self, inner_experiment: Experiment, n_iterations: int):
        self.inner_experiment = inner_experiment
        self.n_iterations = n_iterations

    def execute(self, data):
        return [self.inner_experiment.execute(data) for _ in range(self.n_iterations)]
    