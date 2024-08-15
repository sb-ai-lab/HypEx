from hypex.reporters.matching import MatchingDatasetReporter

from .base import Output
from ..dataset import Dataset, ExperimentData, AdditionalMatchingRole


class MatchingOutput(Output):
    resume: Dataset
    full_data: Dataset

    def __init__(self):
        super().__init__(resume_reporter=MatchingDatasetReporter())

    def _extract_full_data(self, experiment_data: ExperimentData):
        indexes = experiment_data.additional_fields[
            experiment_data.additional_fields.search_columns(AdditionalMatchingRole())[0]
        ]
        indexes.index = experiment_data.ds.index
        filtered_field = indexes.drop(
            indexes[indexes[indexes.columns[0]] == -1], axis=0
        )
        matched_data = experiment_data.ds.loc[
            list(map(lambda x: x[0], filtered_field.get_values()))
        ].rename({i: i + "_matched" for i in experiment_data.ds.columns})
        matched_data.index = filtered_field.index
        self.indexes = indexes
        self.full_data = experiment_data.ds.append(
            matched_data.reindex(experiment_data.ds.index), axis=1
        )

    def extract(self, experiment_data: ExperimentData):
        super().extract(experiment_data)
        self._extract_full_data(experiment_data)
