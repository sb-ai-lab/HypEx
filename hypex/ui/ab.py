from typing import Dict, Union

from .base import Output
from ..analyzers.ab import ABAnalyzer
from ..comparators import GroupSizes, GroupDifference
from ..dataset import ExperimentData, Dataset, TreatmentRole, StatisticRole
from ..reporters.ab import ABDatasetReporter
from ..utils import ExperimentDataEnum, ID_SPLIT_SYMBOL


class ABOutput(Output):
    multitest: Union[Dataset, str]
    all_analysis_results: Dict

    def __init__(self):
        self._groups = []
        super().__init__(resume_reporter=ABDatasetReporter())

    def _extract_multitest_result(self, experiment_data: ExperimentData):
        multitest_id = experiment_data.get_one_id(
            ABAnalyzer, ExperimentDataEnum.analysis_tables
        )
        if multitest_id and "MultiTest" in multitest_id:
            self.multitest = experiment_data.analysis_tables[multitest_id]
        else:
            self.multitest = (
                "There was less than three groups or multitest method wasn't provided"
            )

    def _extract_differences(self, experiment_data: ExperimentData):
        targets = []
        groups = []
        ids = experiment_data.get_ids(
            GroupDifference,
            searched_space=ExperimentDataEnum.analysis_tables,
        )["GroupDifference"]["analysis_tables"]
        self._groups = list(
            experiment_data.groups[
                experiment_data.ds.search_columns(TreatmentRole())[0]
            ].keys()
        )[1:]
        for i in self._groups:
            groups += [i] * len(ids)
        self.difference = Dataset.create_empty()
        for i in range(0, len(ids)):
            self.difference = self.difference.append(
                experiment_data.analysis_tables[ids[i]]
            )
            targets += [ids[i].split(ID_SPLIT_SYMBOL)[-1]]
        self.difference = self.difference.add_column(
            groups, role={"group": StatisticRole()}
        ).add_column(targets * len(self._groups), role={"field": StatisticRole()})

    def _extract_sizes(self, experiment_data: ExperimentData):
        ids = experiment_data.get_ids(
            GroupSizes,
            searched_space=ExperimentDataEnum.analysis_tables,
        )["GroupSizes"]["analysis_tables"]
        self.sizes = experiment_data.analysis_tables[ids[0]].add_column(
            self._groups, role={"group": StatisticRole()}
        )

    def extract(self, experiment_data: ExperimentData):
        super().extract(experiment_data)
        self._extract_multitest_result(experiment_data)
        self._extract_differences(experiment_data)
        self._extract_sizes(experiment_data)
