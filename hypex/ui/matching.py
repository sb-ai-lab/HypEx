from hypex.reporters.matching import MatchingDatasetReporter, MatchingDictReporter

from .base import Output
from ..dataset import Dataset, ExperimentData
from ..utils import ExperimentDataEnum


class MatchingOutput(Output):
    resume: Dataset
    full_data: Dataset

    def __init__(self):
        super().__init__(resume_reporter=MatchingDatasetReporter())

    def _extract_full_data(self, experiment_data: ExperimentData):
        id_ = experiment_data.get_one_id(
            "FaissNearestNeighbors",
            ExperimentDataEnum.groups,
        )
        matched_data = experiment_data.groups[id_]["matched_df"]
        self.full_data = experiment_data.ds.append(matched_data, axis=1)

    def extract(self, experiment_data: ExperimentData):
        super().extract(experiment_data)
        self._extract_full_data(experiment_data)
