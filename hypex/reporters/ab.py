from __future__ import annotations

from typing import Any, ClassVar

from ..analyzers.ab import ABAnalyzer
from ..comparators import Chi2Test, TTest, UTest
from ..dataset import Dataset, ExperimentData, StatisticRole
from ..utils import ExperimentDataEnum
from .aa import OneAADictReporter


class ABDictReporter(OneAADictReporter):
    tests: ClassVar[list] = [TTest, UTest, Chi2Test]

    def extract_analyzer_data(self, data: ExperimentData) -> dict[str, Any]:
        analyzer_id = data.get_one_id(ABAnalyzer, ExperimentDataEnum.analysis_tables)
        return self.extract_from_one_row_dataset(data.analysis_tables[analyzer_id])

    def extract_data_from_analysis_tables(self, data: ExperimentData) -> dict[str, Any]:
        result = {}
        result.update(self.extract_group_sizes(data))
        result.update(self.extract_group_difference(data))
        result.update(self.extract_tests(data))
        result.update(self.extract_analyzer_data(data))
        return result

    def report(self, data: ExperimentData) -> dict[str, Any]:
        return self.extract_data_from_analysis_tables(data)


class ABDatasetReporter(ABDictReporter):
    @staticmethod
    def _invert_aa_format(table: Dataset) -> Dataset:
        return table if table.is_empty() else table.replace("NOT OK", "N").replace("OK", "NOT OK").replace("N", "OK")

    def report_variance_reductions(self, data: ExperimentData) -> Dataset | str:
        """Generate variance reduction report for CUPED/CUPAC transformations."""
        variance_cols = [col for col in data.additional_fields.columns if col.endswith('_variance_reduction')]
        if not variance_cols:
            return "No variance reduction data available. Ensure CUPED or CUPAC was applied."
        
        # Create report data
        report_data = []
        for col in variance_cols:
            metric_name = col.replace('_variance_reduction', '')
            # Get the scalar value from the additional_fields
            reduction_value = data.additional_fields.data[col].iloc[0]
            report_data.append({
                "Transformed Metric Name": metric_name, 
                "Variance Reduction (%)": reduction_value
            })
        
        # Convert to Dataset
        if report_data:
            return Dataset.from_dict(
                data=report_data,
                roles={
                    "Transformed Metric Name": StatisticRole(),
                    "Variance Reduction (%)": StatisticRole()
                }
            )
        return "No variance reduction data available."

    def report(self, data: ExperimentData):
        front_buffer = self.front
        self.front = False
        dict_report = super().report(data)
        self.front = front_buffer
        result = self.convert_flat_dataset(dict_report)
        return self._invert_aa_format(result)
