"""CUPED output class for variance reduction results."""

from ..dataset import Dataset, ExperimentData
from ..reporters.cuped import CupedReporter


class CupedOutput:
    """Output container for CUPED variance reduction analysis.
    
    Attributes:
        variance_reductions: Dataset containing variance reduction metrics
                           for CUPED-transformed targets
    """

    def __init__(self):
        self.variance_reductions: Dataset | str | None = None

    def extract(self, experiment_data: ExperimentData) -> None:
        """Extract CUPED variance reduction data from experiment results.
        
        Args:
            experiment_data: Experiment data containing variance reduction metrics
        """
        self.variance_reductions = CupedReporter.extract_variance_reductions(
            experiment_data
        )

    def __repr__(self) -> str:
        """Return string representation showing available data."""
        if isinstance(self.variance_reductions, Dataset):
            n_metrics = len(self.variance_reductions.data)
            return f"CupedOutput(variance_reductions: {n_metrics} metric(s))"
        elif isinstance(self.variance_reductions, str):
            return f"CupedOutput(variance_reductions: {self.variance_reductions})"
        else:
            return "CupedOutput(variance_reductions: None)"
