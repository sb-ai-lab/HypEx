from __future__ import annotations

from typing import Any

import pandas as pd

from ..dataset import Dataset, ExperimentData, StatisticRole
from ..ml.experiment import ModelSelection
from ..ml.metrics import MLMetric
from ..utils import ExperimentDataEnum
from .abstract import DictReporter, Reporter


class ModelSelectionDictReporter(DictReporter):
    """Extracts ModelSelection results as a flat dict.

    Reads cv_results and final metrics from ExperimentData.variables.
    """

    def _extract_cv_results(self, data: ExperimentData) -> dict[str, Any]:
        """Extract CV results from variables."""
        experiment_id = data.get_one_id(
            ModelSelection, ExperimentDataEnum.variables
        )
        experiment_vars = data.variables[experiment_id]
        cv_results = experiment_vars.get("cv_results", {})
        best_model = experiment_vars.get("best_model", "")

        result: dict[str, Any] = {"best_model": best_model}
        for model_name, metrics in cv_results.items():
            for metric_name, vals in metrics.items():
                prefix = f"{model_name}" if self.front else model_name
                result[f"{prefix} {metric_name} mean"] = vals["mean"]
                result[f"{prefix} {metric_name} std"] = vals["std"]
        return result

    def _extract_final_metrics(self, data: ExperimentData) -> dict[str, Any]:
        """Extract final metric values from variables."""
        result: dict[str, Any] = {}
        metric_ids = data.get_ids(
            MLMetric.__subclasses__(), ExperimentDataEnum.variables
        )
        for class_name, spaces in metric_ids.items():
            for var_id in spaces.get(ExperimentDataEnum.variables.value, []):
                if var_id in data.variables:
                    for key, value in data.variables[var_id].items():
                        result[f"final_{key}"] = value
        return result

    def report(self, data: ExperimentData) -> dict[str, Any]:
        """Return a flat dict with CV stats and final metric values.

        Args:
            data: Experiment data produced by :class:`ModelSelection`.

        Returns:
            Dict with keys like ``'best_model'``,
            ``'<model> <metric> mean'``, ``'final_<metric>'``, etc.
        """
        result = self._extract_cv_results(data)
        result.update(self._extract_final_metrics(data))
        return result


class ModelSelectionReporter(Reporter):
    """Formats ModelSelection results into a readable Dataset.

    Returns a Dataset with columns: model, metric_mean, metric_std, ...
    and a 'best' marker column.
    """

    def report(self, data: ExperimentData) -> Dataset:
        """Build a formatted CV results table from ExperimentData."""
        experiment_id = data.get_one_id(
            ModelSelection, ExperimentDataEnum.variables
        )
        experiment_vars = data.variables[experiment_id]
        cv_results = experiment_vars.get("cv_results", {})
        best_model = experiment_vars.get("best_model", "")

        rows = []
        for model_name, metrics in cv_results.items():
            row: dict[str, Any] = {"model": model_name}
            for metric_name, vals in metrics.items():
                row[f"{metric_name}_mean"] = round(vals["mean"], 6)
                row[f"{metric_name}_std"] = round(vals["std"], 6)
            row["best"] = model_name == best_model
            rows.append(row)

        cv_df = pd.DataFrame(rows)
        return Dataset(
            roles={c: StatisticRole() for c in cv_df.columns},
            data=cv_df,
        )
