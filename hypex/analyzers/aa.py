from typing import Dict, List

from hypex.analyzers.abstract import Analyzer
from hypex.comparators import TTest, KSTest
from hypex.dataset import ExperimentData, Dataset
from hypex.dataset import StatisticRole
from hypex.experiments.base import (
    Executor,
)
from hypex.stats import Mean
from hypex.utils import ExperimentDataEnum, BackendsEnum


class OneAAStatAnalyzer(Analyzer):
    default_inner_executors: Dict[str, Executor] = {"mean": Mean()}

    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.analysis_tables,
            self.id,
            str(self.full_name),
            value,
        )

    @staticmethod
    def _get_test_ids(data: ExperimentData) -> Dict[type, Dict[str, List[str]]]:
        analysis_tests: List[type] = [TTest, KSTest]
        return data.get_ids_by_executors(analysis_tests)

    def execute(self, data: ExperimentData) -> ExperimentData:
        executor_ids = self._get_test_ids(data)
        analysis_data = {}

        mean_operator = self.inner_executors["mean"]
        for c, spaces in executor_ids.items():
            analysis_ids = spaces.get("analysis_tables", [])
            if len(analysis_ids) > 0:
                t_data = data.analysis_tables[analysis_ids[0]]
                for aid in analysis_ids[1:]:
                    t_data = t_data.append(data.analysis_tables[aid])
                t_data.data.index = analysis_ids

                for f in ["p-value", "pass"]:
                    analysis_data[f"{c.__name__} {f}"] = mean_operator.calc(t_data[f])

        analysis_data["mean test score"] = (
            analysis_data["TTest p-value"] + 2 * analysis_data["KSTest p-value"]
        ) / 3
        analysis_data = Dataset.from_dict(
            [analysis_data],
            {f: StatisticRole() for f in analysis_data},
            BackendsEnum.pandas,
        )

        return self._set_value(data, analysis_data)


class OneAAResumeAnalyzer(OneAAStatAnalyzer):
    def execute(self, data: ExperimentData) -> ExperimentData:
        executor_ids = self._get_test_ids(data)

    
