from hypex.analyzers.matching import MatchingAnalyzer
from hypex.dataset.dataset import ExperimentData
from hypex.reporters.abstract import Reporter
from hypex.utils.enums import ExperimentDataEnum


class MatchingDatasetReporter(Reporter):
    def __init__(self, searching_class: type = MatchingAnalyzer):
        self.searching_class = searching_class
        super().__init__()

    def report(self, data: ExperimentData):
        analyzer_id = data.get_one_id(
            self.searching_class, ExperimentDataEnum.analysis_tables
        )
        result = data.analysis_tables[analyzer_id]
        return result
