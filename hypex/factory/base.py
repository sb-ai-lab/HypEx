import sys

from hypex.dataset import (
    ExperimentData,
)
from hypex.experiments import Experiment
from hypex.hypotheses.hypothesis import Hypothesis


class Factory:
    def __init__(self, hypothesis: Hypothesis):
        self.hypothesis = hypothesis

    def make_experiment(self, experiment):
        executors = []
        for key, items in experiment.items():
            class_ = getattr(sys.modules[__name__], key)
            if "executors" in items:
                items["executors"] = self.make_experiment(experiment[key]["executors"])
            if "inner_executors" in items:
                items["inner_executors"] = self.make_experiment(
                    experiment[key]["inner_executors"]
                )
            items = {i: None if j == "None" else j for i, j in items.items()}
            executors.append(class_(**items))
        return executors

    def execute(self):
        experiment_data = ExperimentData(self.hypothesis.dataset)
        experiment = Experiment(
            executors=self.make_experiment(self.hypothesis.experiment)
        )
        return experiment_data, experiment


factory = Factory(
    Hypothesis("C:\\Users\\User\\PycharmProjects\\HypEx\\test_config.json")
)
print(factory.execute())
