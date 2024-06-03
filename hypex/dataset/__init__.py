"""__init__.py for the dataset module in the HypEx library.
This module defines data structures and roles used across the library for managing and manipulating experimental data.
"""

from .abstract import DatasetBase
from .dataset import Dataset, ExperimentData
from .roles import (
    InfoRole,
    TargetRole,
    FeatureRole,
    GroupingRole,
    PreTargetRole,
    StatisticRole,
    StratificationRole,
    TreatmentRole,
    TempTreatmentRole,
    TempGroupingRole,
    TempTargetRole,
    ABCRole,
    default_roles,
)

__all__ = [
    "Dataset",
    "ExperimentData",
    "InfoRole",
    "TargetRole",
    "FeatureRole",
    "GroupingRole",
    "PreTargetRole",
    "StratificationRole",
    "StatisticRole",
    "TreatmentRole",
    "TempTreatmentRole",
    "TempGroupingRole",
    "TempTargetRole",
    "ABCRole",
    "default_roles",
    "DatasetBase",
]
