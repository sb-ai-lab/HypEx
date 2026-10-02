from ..dataset import Dataset, ExperimentData
from ..reporters.homo import HomoDatasetReporter
from .base import Output


class HomoOutput(Output):
    summary: Dataset

    def __init__(self):
        super().__init__(summary_reporter=HomoDatasetReporter())

    def extract(self, experiment_data: ExperimentData):
        super().extract(experiment_data)
