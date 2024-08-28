from hypex.analyzers.matching import MatchingAnalyzer
from hypex.dataset.dataset import ExperimentData
from hypex.dataset.roles import StatisticRole
from hypex.reporters.abstract import Reporter
from hypex.utils.enums import ExperimentDataEnum


class MatchingDatasetReporter(Reporter):

    def report(self, data: ExperimentData):
        analyzer_id = data.get_one_id(
            MatchingAnalyzer, ExperimentDataEnum.analysis_tables
        )
        columns = ["Effect Size", "Standart Error", "P-value", "CI Lower", "CI Upper"]
        result = data.analysis_tables[analyzer_id].transpose(roles={column: StatisticRole() for column in columns}) 
        return result
