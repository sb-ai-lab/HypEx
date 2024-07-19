from hypex.dataset.roles import InfoRole
from hypex.reporters.matching import MatchingDictReporter

from .base import Output
from ..dataset import Dataset, ExperimentData
from ..utils import ExperimentDataEnum


class MatchingOutput(Output):
    resume: Dataset
    full_data: Dataset

    def __init__(self):
        super().__init__(resume_reporter=MatchingDictReporter())

    def _extract_full_data(self, experiment_data: ExperimentData):
        id_ = experiment_data.get_one_id(
            "FaissNearestNeighbors",
            ExperimentDataEnum.groups,
        )
        matched_data = experiment_data.groups[id_]["matched_df"]
        left_on = experiment_data.ds.search_columns(InfoRole())[0]
        right_on = matched_data.search_columns(InfoRole())[0]
        self.full_data = experiment_data.ds.merge(
            matched_data, left_on=left_on, right_on=right_on
        )

    def extract(self, experiment_data: ExperimentData):
        super().extract(experiment_data)
        self._extract_full_data(experiment_data)
