from typing import Dict, List

from hypex.analyzers.abstract import Analyzer
from hypex.comparators import ATE
from hypex.comparators import TTest, UTest
from hypex.dataset import ExperimentData, Dataset
from hypex.dataset import StatisticRole
from hypex.experiments.base import (
    Executor,
)
from hypex.stats import Mean
from hypex.utils import ExperimentDataEnum, BackendsEnum


class ABAnalyzer(Executor):
    default_inner_executors: Dict[str, Executor] = {"mean": Mean()}

    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.analysis_tables,
            self.id,
            str(self.full_name),
            value,
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        analysis_tests: List[type] = [TTest, UTest, ATE]
        executor_ids = data.get_ids(analysis_tests)

        analysis_data = {}
        mean_operator: Mean = self.inner_executors["mean"]
        for c, spaces in executor_ids.items():
            analysis_ids = spaces.get("analysis_tables", [])
            if len(analysis_ids) == 0:
                continue
            t_data = data.analysis_tables[analysis_ids[0]]
            for aid in analysis_ids[1:]:
                t_data = t_data.append(data.analysis_tables[aid])
            t_data.data.index = analysis_ids

            if c.__name__ in ["TTest", "MannWhitney"]:
                for f in ["p-value", "pass"]:
                    analysis_data[f"{c.__name__} {f}"] = mean_operator.calc(
                        t_data[f]
                    ).iloc[0]
            else:
                indexes = t_data.index
                values = t_data.data.values.tolist()
                for idx, value in zip(indexes, values):
                    analysis_data[
                        f"{c.__name__} {idx.split('╰╰')[1].split('[[]')[0]}"
                    ] = value[0]
        analysis_data = Dataset.from_dict(
            [analysis_data],
            {f: StatisticRole() for f in analysis_data},
            BackendsEnum.pandas,
        )

        return self._set_value(data, analysis_data)
