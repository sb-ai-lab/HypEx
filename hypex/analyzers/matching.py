from typing import cast

from ..dataset import SmallDataset
from ..dataset.dataset import DatasetAdapter
from ..dataset.experiment_data import ExperimentData
from ..dataset.roles import StatisticRole
from ..executor.executor import Executor
from ..operators.operators import MatchingMetrics
from ..utils.enums import ExperimentDataEnum


class MatchingAnalyzer(Executor):
    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.analysis_tables, self.id, value, key=key
        )

    def execute(self, data: ExperimentData):
        variables = data.variables[
            data.get_one_id(MatchingMetrics, space=ExperimentDataEnum.variables)
        ]
        columns = ["Effect Size", "Standard Error", "P-value", "CI Lower", "CI Upper"]
        result = cast(
            SmallDataset,
            DatasetAdapter.to_dataset(
                variables,
                {field: StatisticRole(float) for field in list(variables.keys())},
            ),
        )
        transposed = result.transpose(
            roles={column: StatisticRole(float) for column in columns}
        )
        return self._set_value(data, transposed)
