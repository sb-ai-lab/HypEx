"""CUPAC-specific output classes."""
from __future__ import annotations

from ..dataset import Dataset, ExperimentData
from ..reporters.cupac import CupacReporter


class CupacOutput:
    """Container for CUPAC-specific outputs.

    Attributes:
        variance_reductions (Dataset | None): Variance reduction metrics from CUPAC models.
        feature_importances (Dataset | None): Feature importance scores from CUPAC models.
    """

    def __init__(self):
        self.variance_reductions: Dataset | None = None
        self.feature_importances: Dataset | None = None

    def extract(self, experiment_data: ExperimentData) -> None:
        """Extract CUPAC analysis results from experiment data.
        
        Args:
            experiment_data: Experiment data containing CUPAC reports
        """
        self.variance_reductions = CupacReporter.extract_variance_reductions(
            experiment_data
        )
        self.feature_importances = CupacReporter.extract_feature_importances(
            experiment_data
        )

    def __repr__(self) -> str:
        has_vr = self.variance_reductions is not None
        has_fi = self.feature_importances is not None

        if not has_vr and not has_fi:
            return "CupacOutput(no CUPAC data available)"

        parts = []
        if has_vr:
            n_targets = len(self.variance_reductions.data)
            parts.append(f"variance_reductions: {n_targets} target(s)")
        if has_fi:
            n_features = len(self.feature_importances.data)
            parts.append(f"feature_importances: {n_features} feature(s)")

        return f"CupacOutput({', '.join(parts)})"
