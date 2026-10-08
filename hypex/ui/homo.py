from ..dataset import Dataset, ExperimentData
from ..reporters.abstract import DictReporter
from ..reporters.homo import HomogeneityReporter
from .base import Output


class HomoOutput(Output):
    summary: Dataset

    def __init__(self):
        super().__init__(
            summary_reporter=HomogeneityReporter(
                DictReporter(), output_format="dataset"
            )
        )

    def extract(self, experiment_data: ExperimentData):
        super().extract(experiment_data)
