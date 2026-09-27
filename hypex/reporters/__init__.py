"""Reporter classes for formatting HypEx experiment results.

This module provides reporters that extract, format, and present
experiment results from ``ExperimentData`` containers. Reporters
support both dictionary and Dataset output formats.

Public API (star-importable):
    AATestReporter, AADatasetReporter, AAPassedReporter,
    AABestSplitReporter, OneAADictReporter,
    ABTestReporter, ABDictReporter, ABDatasetReporter, CupacReporter,
    HomogeneityReporter, HomoDictReporter, HomoDatasetReporter,
    MatchingReporter, MatchingDictReporter, MatchingDatasetReporter,
    MatchingQualityReporter, MatchingQualityDictReporter,
    MatchingQualityDatasetReporter,
    DatasetReporter, DictReporter, Reporter, ResultKey, TestDictReporter,
    REPORTABLE_METRICS.
"""
from __future__ import annotations

from .aa import (
    AABestSplitReporter,
    AADatasetReporter,
    AAPassedReporter,
    AATestReporter,
    OneAADictReporter,
)
from .ab import (
    ABDatasetReporter,
    ABDictReporter,
    ABTestReporter,
    CupacReporter,
)
from .abstract import (
    REPORTABLE_METRICS,
    DatasetReporter,
    DictReporter,
    Reporter,
    ResultKey,
    TestDictReporter,
)
from .homo import (
    HomoDatasetReporter,
    HomoDictReporter,
    HomogeneityReporter,
)
from .matching import (
    MatchingDatasetReporter,
    MatchingDictReporter,
    MatchingQualityDatasetReporter,
    MatchingQualityDictReporter,
    MatchingQualityReporter,
    MatchingReporter,
)

__all__ = [
    "REPORTABLE_METRICS",
    # AA reporters
    "AABestSplitReporter",
    "AADatasetReporter",
    "AAPassedReporter",
    "AATestReporter",
    # AB reporters
    "ABDatasetReporter",
    "ABDictReporter",
    "ABTestReporter",
    "CupacReporter",
    # Abstract / base reporters
    "DatasetReporter",
    "DictReporter",
    # Homogeneity reporters
    "HomoDatasetReporter",
    "HomoDictReporter",
    "HomogeneityReporter",
    # Matching reporters
    "MatchingDatasetReporter",
    "MatchingDictReporter",
    "MatchingQualityDatasetReporter",
    "MatchingQualityDictReporter",
    "MatchingQualityReporter",
    "MatchingReporter",
    "OneAADictReporter",
    "Reporter",
    "ResultKey",
    "TestDictReporter",
]