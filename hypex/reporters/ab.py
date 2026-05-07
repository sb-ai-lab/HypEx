from __future__ import annotations
from typing import Any, ClassVar
import warnings

from ..analyzers.ab import ABAnalyzer
from ..comparators import GroupChi2Test, GroupTTest, GroupUTest
from ..dataset import Dataset, ExperimentData, StatisticRole
from ..utils import ExperimentDataEnum
from .abstract import (
    DictReporter, DatasetReporter, Reporter, 
    extract_group_difference, extract_tests, extract_analyzer_data, extract_group_sizes
)

class ABTestReporter(DatasetReporter):
    """Phase 2: Unified AB reporter."""
    tests: ClassVar[list] = [GroupTTest, GroupUTest, GroupChi2Test]

    def _report(self, data: ExperimentData) -> dict[str, Any]:
        result = {}
        result.update(extract_group_sizes(data, self.front))
        result.update(extract_group_difference(data, self.front))
        result.update(extract_tests(data, self.tests, self.front))
        result.update(extract_analyzer_data(data, ABAnalyzer))
        return result

    def report(self, data: ExperimentData) -> dict | Dataset:
        return self._with_front(data, front_flag=False, func=lambda d: super().report(d))

    @staticmethod
    def report_variance_reductions(data: ExperimentData) -> Dataset | str:
        variance_cols = [c for c in data.additional_fields.columns if c.endswith("_variance_reduction")]
        if not variance_cols:
            return "No variance reduction data available. Ensure CUPED or CUPAC was applied."
        report_data = []
        for col in variance_cols:
            metric_name = col.replace("_variance_reduction", "")
            reduction_value = data.additional_fields.data[col].iloc[0]
            report_data.append({"Transformed Metric Name": metric_name, "Variance Reduction (%)": reduction_value})
        return Dataset.from_dict(
            data=report_data,
            roles={"Transformed Metric Name": StatisticRole(), "Variance Reduction (%)": StatisticRole()},
        ) if report_data else "No variance reduction data available."

class ABDictReporter(ABTestReporter):
    def __init__(self, front=True):
        super().__init__(DictReporter(front=front), output_format="dict")
        warnings.warn("ABDictReporter is deprecated.", DeprecationWarning, stacklevel=2)

class ABDatasetReporter(ABTestReporter):
    def __init__(self):
        super().__init__(DictReporter(), output_format="dataset")
        warnings.warn("ABDatasetReporter is deprecated.", DeprecationWarning, stacklevel=2)

class CupacReporter(Reporter):
    def report(self, data: ExperimentData) -> dict[str, Dataset | None]:
        cupac_keys = [k for k in data.analysis_tables.keys() if k.endswith("_cupac_report")]
        if not cupac_keys:
            return {"variance_reductions": None, "feature_importances": None}

        var_data, imp_data = [], []
        for key in cupac_keys:
            report = data.analysis_tables[key]
            target = key.replace("_cupac_report", "")
            var_data.append({
                "target": target, "best_model": report.get("cupac_best_model"),
                "variance_reduction_cv": report.get("cupac_variance_reduction_cv"),
                "variance_reduction_real": report.get("cupac_variance_reduction_real"),
            })
            for feat, imp in report.get("cupac_feature_importances", {}).items():
                imp_data.append({"target": target, "feature": feat, "importance": imp, "model": report.get("cupac_best_model")})

        vr_ds = Dataset.from_dict(data=var_data, roles={"target": StatisticRole(), "best_model": StatisticRole(), "variance_reduction_cv": StatisticRole(), "variance_reduction_real": StatisticRole()}) if var_data else None
        fi_ds = Dataset.from_dict(data=imp_data, roles={"target": StatisticRole(), "feature": StatisticRole(), "importance": StatisticRole(), "model": StatisticRole()}) if imp_data else None
        return {"variance_reductions": vr_ds, "feature_importances": fi_ds}