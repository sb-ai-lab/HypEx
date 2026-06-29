from ..dataset.dataset import DatasetAdapter
from ..dataset.experiment_data import ExperimentData
from ..dataset.roles import StatisticRole
from ..executor.executor import Executor
from ..operators.operators import MatchingMetrics
from ..utils.enums import ExperimentDataEnum


class MatchingAnalyzer(Executor):
    """Analyzer for matching metrics.

    Retrieves raw matching metrics computed by the MatchingMetrics
    operator, formats them into a structured transposed Dataset,
    and stores the result in the experiment's analysis tables.
    """
    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        """Store the analyzed metrics in the experiment data container.

        Args:
            data: The experiment data container to update.
            value: The Dataset containing the formatted matching metrics.
            key: Optional key for the stored value.

        Returns:
            The updated ExperimentData instance with the metrics stored
            in the analysis_tables space.
        """
        return data.set_value(
            ExperimentDataEnum.analysis_tables, self.id, value, key=key
        )

    def execute(self, data: ExperimentData):
        """Execute the matching metrics analysis pipeline.

        Fetches the raw metric variables computed by MatchingMetrics,
        converts them into a Dataset, transposes the data to align
        with standard metric columns (Effect Size, Standard Error,
        P-value, CI Lower, CI Upper), ensures numeric types, and
        saves the final table to the analysis tables.

        Args:
            data: The experiment data container holding the raw
                matching metrics in its variables space.

        Returns:
            The updated ExperimentData instance with the formatted
            matching metrics stored in the analysis_tables space.
        """
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
