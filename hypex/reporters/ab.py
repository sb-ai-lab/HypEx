from __future__ import annotations
from typing import Any, ClassVar
import warnings

from ..analyzers.ab import ABAnalyzer
from ..comparators import GroupChi2Test, GroupTTest, GroupUTest, BaseComparator
from ..comparators import StatsTTest, StatsChi2Test
from ..dataset import Dataset, ExperimentData, StatisticRole
from .abstract import (
    DictReporter, DatasetReporter, Reporter, 
    extract_group_difference, extract_tests, extract_analyzer_data, extract_group_sizes
)

class ABTestReporter(DatasetReporter):
    """Reporter for A/B test results.

    Extracts group sizes, metric differences, statistical test outcomes,
    and analyzer data, formatting them into a structured dataset or dictionary.
    """
    
    tests: ClassVar[list[type[BaseComparator]]] = [GroupTTest, GroupUTest, GroupChi2Test, StatsTTest, StatsChi2Test]

    def _report(self, data: ExperimentData) -> dict[str, Any]:
        """Construct the internal dictionary report for A/B tests.

        Args:
            data: The experiment data container.

        Returns:
            A dictionary containing group sizes, differences, test results,
            and analyzer metrics.
        """
        result = {}
        result.update(extract_group_sizes(data, self.front))
        result.update(extract_group_difference(data, self.front))
        result.update(extract_tests(data, self.tests, self.front))
        result.update(extract_analyzer_data(data, ABAnalyzer))
        return result

    def _report(self, data: ExperimentData) -> dict[str, Any]:
        """Generate the final A/B test report.

        Ensures the ``front`` formatting flag is disabled before generating the report, 
        then returns the result in the configured format.

        Args:
            data: The experiment data container.

        Returns:
            The report as a dictionary or ``Dataset``.
        """
        result = {}
        result.update(extract_group_sizes(data, self.front))
        result.update(extract_group_difference(data, self.front))
        result.update(extract_tests(data, self.tests, self.front))
        result.update(extract_analyzer_data(data, ABAnalyzer))
        return result

    @staticmethod
    def report_variance_reductions(data: ExperimentData) -> Dataset | str:
        """Extract and format variance reduction metrics from CUPED/CUPAC.

        Args:
            data: The experiment data container.

        Returns:
            A ``Dataset`` with transformed metric names and variance reduction percentages, 
            or a descriptive string if no data is available.
        """
        
        variance_cols = [c for c in data.additional_fields.columns if c.endswith("_variance_reduction")]
        if not variance_cols:
            return "No variance reduction data available. Ensure CUPED or CUPAC was applied."
        
        report_data = []
        records = data.additional_fields.limit(1).to_records()
        first_row = records[0] if records else {}
        
        for col in variance_cols:
            metric_name = col.replace("_variance_reduction", "")
            reduction_value = first_row.get(col)
            report_data.append({
                "Transformed Metric Name": metric_name, 
                "Variance Reduction (%)": reduction_value
            })
        
        return Dataset.from_dict(
            data=report_data,
            roles={
                "Transformed Metric Name": StatisticRole(), 
                "Variance Reduction (%)": StatisticRole()
            },
        ) if report_data else "No variance reduction data available."

class ABDictReporter(ABTestReporter):
    """Legacy reporter wrapper for dictionary output.

    Deprecated: Use ``ABTestReporter(output_format='dict')`` instead.
    """
    def __init__(self, front: bool = True):
        """Initialize the legacy dictionary reporter.

        Args:
            front: If ``True``, formats keys for front-end display.
                Defaults to ``True``.
        """
        super().__init__(DictReporter(front=front), output_format="dict")
        warnings.warn("ABDictReporter is deprecated.", DeprecationWarning, stacklevel=2)

class ABDatasetReporter(ABTestReporter):
    """Legacy reporter wrapper for dataset output.

    Deprecated: Use ``ABTestReporter()`` instead.
    """
    def __init__(self):
        """Initialize the legacy dataset reporter."""
        super().__init__(DictReporter(), output_format="dataset")
        warnings.warn("ABDatasetReporter is deprecated.", DeprecationWarning, stacklevel=2)

class CupacReporter(Reporter):
    """Reporter for CUPAC variance reduction results.

    Extracts variance reduction metrics and feature importances from
    CUPAC model reports stored in the experiment data.
    """
    def report(self, data: ExperimentData) -> dict[str, Dataset | None]:
        """Generate a CUPAC results report.

        Args:
            data: The experiment data container.

        Returns:
            A dictionary containing ``'variance_reductions'`` and
            ``'feature_importances'`` as ``Dataset`` instances, or
            ``None`` if no CUPAC data is found.
        """
        cupac_keys = [k for k in data.analysis_tables.keys() if k.endswith("_cupac_report")]
        if not cupac_keys:
            return {"variance_reductions": None, "feature_importances": None}

        var_data, imp_data = [], []
        for key in cupac_keys:
            report = data.analysis_tables[key]
            target = key.replace("_cupac_report", "")
            
            if isinstance(report, dict):
                get_val = report.get
                imp_dict = report.get("cupac_feature_importances", {})
            else:
                rec = report.to_records()[0] if not report.is_empty() else {}
                get_val = rec.get
                imp_dict = rec.get("cupac_feature_importances", {})

            var_data.append({
                "target": target, 
                "best_model": get_val("cupac_best_model"),
                "variance_reduction_cv": get_val("cupac_variance_reduction_cv"),
                "variance_reduction_real": get_val("cupac_variance_reduction_real"),
            })
            
            if isinstance(imp_dict, dict):
                for feat, imp in imp_dict.items():
                    imp_data.append({
                        "target": target, 
                        "feature": feat, 
                        "importance": imp, 
                        "model": get_val("cupac_best_model")
                    })

        vr_ds = Dataset.from_dict(
            data=var_data, 
            roles={
                "target": StatisticRole(), 
                "best_model": StatisticRole(), 
                "variance_reduction_cv": StatisticRole(), 
                "variance_reduction_real": StatisticRole()
            }
        ) if var_data else None
        
        fi_ds = Dataset.from_dict(
            data=imp_data, 
            roles={
                "target": StatisticRole(), 
                "feature": StatisticRole(), 
                "importance": StatisticRole(), 
                "model": StatisticRole()
            }
        ) if imp_data else None
        
        return {"variance_reductions": vr_ds, "feature_importances": fi_ds}