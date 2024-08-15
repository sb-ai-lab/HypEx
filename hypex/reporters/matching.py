from hypex.analyzers.matching import MatchingAnalyzer
from hypex.dataset.dataset import ExperimentData
from hypex.reporters.abstract import DatasetReporter, DictReporter
from hypex.utils.enums import ExperimentDataEnum


class MatchingDictReporter(DictReporter):

    def _extract_from_analyser(self, data: ExperimentData):
        analyzer_id = data.get_one_id(
            MatchingAnalyzer, ExperimentDataEnum.analysis_tables
        )
        return self.extract_from_one_row_dataset(data.analysis_tables[analyzer_id])

    def report(self, experiment_data: ExperimentData):
        result = {}
        result.update(self._extract_from_analyser(experiment_data))
        return result


class MatchingDatasetReporter(DatasetReporter):
    def __init__(self) -> None:
        self.dict_reporter = MatchingDictReporter()
