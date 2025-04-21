from ..dataset.dataset import DatasetAdapter, ExperimentData
from ..dataset.roles import StatisticRole
from ..operators.operators import MatchingMetrics
from ..utils.enums import ExperimentDataEnum
from .base import Analyzer


class MatchingAnalyzer(Analyzer):
    def execute(self, data: ExperimentData):
        variables = data.variables[
            data.get_one_id(MatchingMetrics, space=ExperimentDataEnum.variables)
        ]
        columns = ["Effect Size", "Standard Error", "P-value", "CI Lower", "CI Upper"]
        return self._set_value(
            data,
            DatasetAdapter.to_dataset(
                variables,
                {field: StatisticRole() for field in list(variables.keys())},
            ).transpose(roles={column: StatisticRole() for column in columns}),
        )
