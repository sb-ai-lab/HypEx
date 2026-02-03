"""CUPAC-specific output classes."""
from __future__ import annotations

from ..dataset import Dataset, ExperimentData
from ..reporters.cupac import CupacReporter
from .base import Output


class CupacOutput(Output):
    """Container for CUPAC-specific outputs.

    Attributes:
        resume (Dataset | None): Summary of CUPAC results (target, best_model, variance_reduction_cv, variance_reduction_real).
        variance_reductions (Dataset | None): Variance reduction metrics from CUPAC models.
        feature_importances (Dataset | None): Feature importance scores from CUPAC models.
    """

    variance_reductions: Dataset | None
    feature_importances: Dataset | None

    def __init__(self):
        super().__init__(resume_reporter=CupacReporter())
        self.variance_reductions = None
        self.feature_importances = None

    def extract(self, experiment_data: ExperimentData) -> None:
        """Extract CUPAC analysis results from experiment data.
        
        Args:
            experiment_data: Experiment data containing CUPAC reports
        """
        self._extract_by_reporters(experiment_data)
        self.variance_reductions = CupacReporter.extract_variance_reductions(
            experiment_data
        )
        self.feature_importances = CupacReporter.extract_feature_importances(
            experiment_data
        )
