from typing import List, Optional, Any

from hypex.comparators import ATE, TTest, UTest
from hypex.dataset import Dataset, ExperimentData, StatisticRole
from hypex.experiments.base import Executor
from hypex.utils import (
    ID_SPLIT_SYMBOL,
    NAME_BORDER_SYMBOL,
    BackendsEnum,
    ExperimentDataEnum,
    ABNTestMethodsTypes,
)


class ABAnalyzer(Executor):

    def __init__(
        self,
        multitest_method: Optional[ABNTestMethodsTypes] = None,
        full_name: Optional[str] = None,
        key: Any = "",
    ):
        self.multitest_method = multitest_method
        super().__init__(full_name, key)

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
        multitest_pvalues = []
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
                    value = t_data[f]
                    if (
                        c.__name__ == "TTest"
                        and self.multitest_method
                        and f == "p-value"
                    ):
                        multitest_pvalues.append(value.data)
                    analysis_data[f"{c.__name__} {f}"] = value.mean()

            else:
                indexes = t_data.index
                values = t_data.data.values.tolist()
                for idx, value in zip(indexes, values):
                    name = idx.split(ID_SPLIT_SYMBOL)[-1]
                    analysis_data[
                        f"{c.__name__} {name[name.find(NAME_BORDER_SYMBOL) + 1: name.rfind(NAME_BORDER_SYMBOL)]}"
                    ] = value[0]
        analysis_dataset: Dataset = Dataset.from_dict(
            [analysis_data],
            {f: StatisticRole() for f in analysis_data},
            BackendsEnum.pandas,
        )

        return self._set_value(data, analysis_dataset)
