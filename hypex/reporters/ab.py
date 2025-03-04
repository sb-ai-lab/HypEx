from typing import Dict, Any

from ..analyzers.ab import ABAnalyzer
from ..dataset import ExperimentData, Dataset
from ..utils import ExperimentDataEnum
from .aa import OneAADictReporter
from ..comparators import TTest, UTest, Chi2Test


class ABDictReporter(OneAADictReporter):
    tests = [TTest, UTest, Chi2Test]

    def extract_analyzer_data(self, data: ExperimentData) -> Dict[str, Any]:
        analyzer_id = data.get_one_id(ABAnalyzer, ExperimentDataEnum.analysis_tables)
        return self.extract_from_one_row_dataset(data.analysis_tables[analyzer_id])

    def extract_data_from_analysis_tables(self, data: ExperimentData) -> Dict[str, Any]:
        result = {}
        result.update(self.extract_group_sizes(data))
        result.update(self.extract_group_difference(data))
        result.update(self.extract_tests(data))
        result.update(self.extract_analyzer_data(data))
        return result

    def report(self, data: ExperimentData) -> Dict[str, Any]:
        return self.extract_data_from_analysis_tables(data)


class ABDatasetReporter(ABDictReporter):
    @staticmethod
    def _invert_aa_format(table: Dataset) -> Dataset:
        return table.replace("NOT OK", "N").replace("OK", "NOT OK").replace("N", "OK")

    def report(self, data: ExperimentData):
        front_buffer = self.front
        self.front = False
        dict_report = super().report(data)
        self.front = front_buffer
        result = self.convert_flat_dataset(dict_report)
        return self._invert_aa_format(result)
