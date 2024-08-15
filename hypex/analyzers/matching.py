from hypex.dataset.dataset import DatasetAdapter, ExperimentData
from hypex.dataset.roles import StatisticRole
from hypex.executor.executor import Executor
from hypex.operators.operators import MatchingMetrics
from hypex.utils.enums import ExperimentDataEnum


class MatchingAnalyzer(Executor):
    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.analysis_tables, self.id, value, key=key
        )

    def execute(self, data: ExperimentData):
        variables = data.variables[
            data.get_one_id(MatchingMetrics, space=ExperimentDataEnum.variables)
        ]
        return self._set_value(
            data,
            DatasetAdapter.to_dataset(
                variables,
                {field: StatisticRole() for field in list(variables.keys())},
            ),
        )

    # здесь будет анализ всех тестов
