from __future__ import annotations

from typing import Any

from ..dataset import (
    AdditionalMatchingRole,
    Dataset,
    ExperimentData,
    TargetRole,
)
from ..reporters.matching import MatchingReporter, MatchingQualityReporter
from .base import Output
from ..utils.adapter import Adapter


class MatchingOutput(Output):
    """Output handler for Matching experiments.

    Automatically extracts and formats the results of a matching pipeline,
    including core treatment effect metrics, quality test results, and the
    fully enriched dataset with matched pairs.

    Attributes:
        resume: A Dataset containing the core matching metrics (ATT, ATC, ATE).
        full_data: The original dataset enriched with matched features
            (e.g., columns suffixed with ``_matched_0``, ``_matched_1``).
        quality_results: A dictionary mapping feature names to Datasets
            containing quality test results (T-Test, Chi2, KS) per feature.
    """
    resume: Dataset
    full_data: Dataset
    quality_results: dict[str, Dataset]

    def __init__(self, *args, **kwargs):
        """Initializes the MatchingOutput handler.

        Configures the output with a ``MatchingReporter`` for the main resume
        and a ``MatchingQualityReporter`` for the quality tests.

        Args:
            *args: Positional arguments passed to the base ``Output`` class.
            **kwargs: Keyword arguments passed to the base ``Output`` class.
        """
        super().__init__(
            resume_reporter=MatchingReporter(),
            additional_reporters={
                "quality_results": MatchingQualityReporter()
            },
        )

    def extract(self, experiment_data: ExperimentData):
        """Extracts and formats all matching experiment results.

        Calls the base extraction logic, rounds the main resume metrics to
        4 decimal places, and enriches the original dataset with matched pairs.

        Args:
            experiment_data: The experiment data container holding the results
                of the matching pipeline.
        """
        super().extract(experiment_data)
        
        if self.resume is not None and not self.resume.is_empty():
            if self.resume is not None and not self.resume.is_empty():
                self.resume = round(self.resume, 4)
            
        self._extract_full_data(experiment_data)

    def _extract_full_data(self, experiment_data: ExperimentData):
        """Enriches the original dataset with matched pair data.

        Retrieves the match indices stored in the additional fields (where
        ``FaissNearestNeighbors`` saves the matched pair indices with the
        ``AdditionalMatchingRole``) and appends the corresponding matched
        rows to the main dataset. Matched columns are suffixed with their
        match index (e.g., ``_matched_0``).

        Args:
            experiment_data: The experiment data container holding the match
                indices and the original dataset.
        """
        self.full_data = experiment_data.ds
        
        index_cols = experiment_data.ds.search_columns(AdditionalMatchingRole())
        if not index_cols:
            return

        for i, col in enumerate(sorted(index_cols)):
            t_indexes = experiment_data.ds[[col]]
            
            idx_values = t_indexes[col].get_values()
            orig_indices = Adapter.to_list(t_indexes.index) 
            
            valid_orig_indices = []
            valid_matched_indices = []
            
            for orig_idx, matched_idx in zip(orig_indices, idx_values):
                if matched_idx != -1 and matched_idx is not None and str(matched_idx) != 'nan':
                    valid_orig_indices.append(orig_idx)
                    valid_matched_indices.append(matched_idx)
            
            if not valid_matched_indices:
                continue
                
            try:
                matched_data = experiment_data.ds.loc[valid_matched_indices]
                
                rename_dict = {c: f"{c}_matched_{i}" for c in experiment_data.ds.columns}
                matched_data = matched_data.rename(rename_dict)
                
                matched_data.index = valid_orig_indices
                
                matched_data = matched_data.reindex(experiment_data.ds.index)
                
                self.full_data = self.full_data.append(matched_data, axis=1)
            except Exception:
                continue