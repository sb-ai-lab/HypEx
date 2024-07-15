from typing import Dict

from .base import Output
from ..analyzers.ab import ABAnalyzer
from ..dataset import ExperimentData, Dataset
from ..reporters.ab import ABDatasetReporter
from ..utils import ExperimentDataEnum


class ABOutput(Output):
    multitest_result: Dataset
    all_analysis_results: Dict

    def __init__(self):
        super().__init__(resume_reporter=ABDatasetReporter())

    def _extract_multitest_result(self, experiment_data: ExperimentData):
        multitest_id = experiment_data.get_one_id(
            ABAnalyzer, ExperimentDataEnum.analysis_tables
        )
        if multitest_id:
            self.multitest_result = experiment_data.analysis_tables[multitest_id]

    def _set_all_analysis_results(self, experiment_data: ExperimentData):
        self.all_analysis_results = experiment_data.analysis_tables

    def extract(self, experiment_data: ExperimentData):
        super().extract(experiment_data)
        self._extract_multitest_result(experiment_data)
        self._set_all_analysis_results(experiment_data)
