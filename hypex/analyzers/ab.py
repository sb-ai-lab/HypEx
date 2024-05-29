from typing import List

from hypex.comparators import ATE, TTest, UTest
from hypex.dataset import Dataset, ExperimentData, StatisticRole
from hypex.experiments.base import Executor
from hypex.utils import (
    ID_SPLIT_SYMBOL,
    NAME_BORDER_SYMBOL,
    BackendsEnum,
    ExperimentDataEnum,
)


class ABAnalyzer(Executor):
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
        for c, spaces in executor_ids.items():
            analysis_ids = spaces.get("analysis_tables", [])
            if len(analysis_ids) == 0:
                continue
            t_data = data.analysis_tables[analysis_ids[0]]
            for aid in analysis_ids[1:]:
                t_data = t_data.append(data.analysis_tables[aid])
            t_data.data.index = analysis_ids

            if c.__name__ in ["TTest", "UTest"]:
                for f in ["p-value", "pass"]:
                    analysis_data[f"{c.__name__} {f}"] = t_data[f].mean()
            else:
                indexes = t_data.index
                values = t_data.data.values.tolist()
                for idx, value in zip(indexes, values):
                    name = idx.split(ID_SPLIT_SYMBOL)[-1]
                    analysis_data[
                        f"{c.__name__} {name[name.find(NAME_BORDER_SYMBOL) + 1: name.rfind(NAME_BORDER_SYMBOL)]}"
                    ] = value[0]
        analysis_data = Dataset.from_dict(
            [analysis_data],
            {f: StatisticRole() for f in analysis_data},
            BackendsEnum.pandas,
        )

        return self._set_value(data, analysis_data)
