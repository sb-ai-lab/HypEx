from ..reporters.homo import HomoDatasetReporter

from .base import Output
from ..dataset import Dataset, ExperimentData


class HomoOutput(Output):
    resume: Dataset

    def __init__(self):
        super().__init__(resume_reporter=HomoDatasetReporter())

    def extract(self, experiment_data: ExperimentData):
        super().extract(experiment_data)
