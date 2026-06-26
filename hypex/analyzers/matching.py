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
        columns = ["Effect Size", "Standard Error", "P-value", "CI Lower", "CI Upper"]
        metrics_id = data.get_one_id(MatchingMetrics, space=ExperimentDataEnum.variables)
        variables = data.variables[metrics_id]
        
        ds_before_transpose = DatasetAdapter.to_dataset(
            variables,
            {field: StatisticRole(float) for field in list(variables.keys())},
        )
        
        ds_after_transpose = ds_before_transpose.transpose(
            roles={column: StatisticRole(float) for column in columns}
        )
        ds_after_transpose = ds_after_transpose.astype({col: float for col in ds_after_transpose.columns})
        
        return self._set_value(data, ds_after_transpose)
