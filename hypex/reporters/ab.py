from typing import Dict, Any

from hypex.analyzers import ABAnalyzer
from hypex.dataset import ExperimentData
from hypex.utils import ExperimentDataEnum
from .aa import AADictReporter


class ABDictReporter(AADictReporter):
    def extract_analyzer_data(self, data: ExperimentData) -> Dict[str, Any]:
        analyzer_id = data._get_one_id(ABAnalyzer, ExperimentDataEnum.analysis_tables)
        return self.extract_from_one_row_dataset(data.analysis_tables[analyzer_id])

    def extract_data_from_analysis_tables(self, data: ExperimentData) -> Dict[str, Any]:
        result = {}
        result.update(self.extract_group_sizes(data))
        result.update(self.extract_group_difference(data))
        result.update(self.extract_analyzer_data(data))
        return result

    def report(self, data: ExperimentData) -> Dict[str, Any]:
        result = {}
        result.update(self.extract_data_from_analysis_tables(data))
        return result
