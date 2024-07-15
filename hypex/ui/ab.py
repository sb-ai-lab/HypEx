from typing import Dict

from .base import Output
from ..analyzers.ab import ABAnalyzer
from ..comparators import GroupSizes, GroupDifference
from ..dataset import ExperimentData, Dataset
from ..reporters.ab import ABDatasetReporter
from ..utils import ExperimentDataEnum, ID_SPLIT_SYMBOL


class ABOutput(Output):
    multitest: Dataset
    all_analysis_results: Dict

    def __init__(self):
        super().__init__(resume_reporter=ABDatasetReporter())

    def _extract_multitest_result(self, experiment_data: ExperimentData):
        multitest_id = experiment_data.get_one_id(
            ABAnalyzer, ExperimentDataEnum.analysis_tables
        )
        if multitest_id and "MultiTest" in multitest_id:
            self.multitest = experiment_data.analysis_tables[multitest_id]

    def _extract_sizes_and_differences(self, experiment_data: ExperimentData):
        ids = experiment_data.get_ids(
            [GroupSizes, GroupDifference],
            searched_space=ExperimentDataEnum.analysis_tables,
        )
        ids = (
            ids["GroupSizes"]["analysis_tables"]
            + ids["GroupDifference"]["analysis_tables"]
        )
        self.difference = experiment_data.analysis_tables[ids[0]].transpose()
        for id_ in range(1, len(ids)):
            data = experiment_data.analysis_tables[ids[id_]].rename(
                {
                    i: f"{ids[id_].split(ID_SPLIT_SYMBOL)[2]} {i}"
                    for i in experiment_data.analysis_tables[ids[id_]].columns
                }
            )
            self.difference = self.difference.append(data.transpose())

    def extract(self, experiment_data: ExperimentData):
        super().extract(experiment_data)
        self._extract_multitest_result(experiment_data)
        self._extract_sizes_and_differences(experiment_data)
