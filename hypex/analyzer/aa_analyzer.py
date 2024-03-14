from typing import List

from hypex.experiment.experiment import Experiment, Executor
from hypex.dataset.dataset import Dataset, ExperimentData
from hypex.dataset.roles import TargetRole

class AATestAnalyzer(Experiment):
    def __init__(self, full_name: str = None, index=0):
        super().__init__([], False, full_name, index)

    # TODO: Implement this method to return a list of executors for the experiment.
    def build(self, data:ExperimentData) -> List[Executor]:
        target_fields = data.data.get_columns_by_roles(TargetRole)

    def execute(self, data: ExperimentData) -> ExperimentData:
        pass