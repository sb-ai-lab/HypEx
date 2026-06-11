"""CUPED output class for variance reduction results."""

from ..dataset import Dataset, ExperimentData
from ..reporters.cuped import CupedReporter
from .base import Output


class CupedOutput(Output):
    """Output container for CUPED variance reduction analysis.
    
    Attributes:
        resume: Dataset with summary of CUPED results (target, covariate, variance_reduction)
        variance_reductions: Dataset containing variance reduction metrics
                           for CUPED-transformed targets
    """

    variance_reductions: Dataset | str | None

    def __init__(self):
        super().__init__(resume_reporter=CupedReporter())
        self.variance_reductions = None

    def extract(self, experiment_data: ExperimentData) -> None:
        """Extract CUPED variance reduction data from experiment results.
        
        Args:
            experiment_data: Experiment data containing variance reduction metrics
        """
        self._extract_by_reporters(experiment_data)
        self.variance_reductions = CupedReporter.extract_variance_reductions(
            experiment_data
        )
