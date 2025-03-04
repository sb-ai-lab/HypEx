"""__init__.py for the dataset module in the HypEx library.
This module defines data structures and roles used across the library for managing and manipulating experimental data.
"""

from .abstract import DatasetBase
from .dataset import Dataset, DatasetAdapter, ExperimentData
from .roles import (
    ABCRole,
    AdditionalGroupingRole,
    AdditionalMatchingRole,
    AdditionalPreTargetRole,
    AdditionalTargetRole,
    AdditionalTreatmentRole,
    DefaultRole,
    FeatureRole,
    FilterRole,
    GroupingRole,
    InfoRole,
    PreTargetRole,
    StatisticRole,
    StratificationRole,
    TargetRole,
    TempGroupingRole,
    TempRole,
    TempTargetRole,
    TempTreatmentRole,
    TreatmentRole,
    default_roles,
)

__all__ = [
    "ABCRole",
    "AdditionalGroupingRole",
    "AdditionalMatchingRole",
    "AdditionalPreTargetRole",
    "AdditionalTargetRole",
    "AdditionalTreatmentRole",
    "Dataset",
    "DatasetAdapter",
    "DatasetBase",
    "DefaultRole",
    "ExperimentData",
    "FeatureRole",
    "FilterRole",
    "GroupingRole",
    "InfoRole",
    "PreTargetRole",
    "StatisticRole",
    "StratificationRole",
    "TargetRole",
    "TempGroupingRole",
    "TempRole",
    "TempTargetRole",
    "TempTreatmentRole",
    "TreatmentRole",
    "default_roles",
]
