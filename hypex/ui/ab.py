from typing import Union

from ..analyzers.ab import ABAnalyzer
from ..comparators import GroupDifference, GroupSizes
from ..dataset import Dataset, ExperimentData, StatisticRole, TreatmentRole
from ..reporters.ab import ABDatasetReporter
from ..utils import ID_SPLIT_SYMBOL, ExperimentDataEnum
from .base import Output


class ABOutput(Output):
    multitest: Union[Dataset, str]
    sizes: Dataset
    variance_reductions: Dataset | None

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
        diff = Dataset.create_empty()
        for i in range(len(ids)):
            diff = diff.append(experiment_data.analysis_tables[ids[i]])
            targets += [ids[i].split(ID_SPLIT_SYMBOL)[-1]]
        return diff.add_column(groups, role={"group": StatisticRole()}).add_column(
            targets * len(self._groups), role={"feature": StatisticRole()}
        )

    def _extract_sizes(self, experiment_data: ExperimentData):
        ids = experiment_data.get_ids(
            GroupSizes,
            searched_space=ExperimentDataEnum.analysis_tables,
        )["GroupSizes"]["analysis_tables"]
        self.sizes = experiment_data.analysis_tables[ids[0]].add_column(
            self._groups, role={"group": StatisticRole()}
        )

    def _extract_variance_reductions(self, experiment_data: ExperimentData):
        """Extract variance reduction data from additional_fields."""
        variance_cols = [
            col
            for col in experiment_data.additional_fields.columns
            if col.endswith("_variance_reduction")
        ]
        if variance_cols:
            self.variance_reductions = experiment_data.additional_fields[variance_cols]
        else:
            self.variance_reductions = None

    @property
    def variance_reduction_report(self) -> Dataset | str:
        """Get variance reduction report for CUPED/CUPAC transformations."""
        if hasattr(self, "_experiment_data"):
            return self.resume_reporter.report_variance_reductions(
                self._experiment_data
            )
        return "No experiment data available."

    def extract(self, experiment_data: ExperimentData):
        super().extract(experiment_data)
        self._extract_differences(experiment_data)
        self._extract_multitest_result(experiment_data)
        self._extract_sizes(experiment_data)
        self._extract_variance_reductions(experiment_data)
