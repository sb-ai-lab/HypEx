"""__init__.py for the dataset module in the HypEx library.
This module defines data structures and roles used across the library for managing and manipulating experimental data.
"""
from hypex.dataset.dataset import Dataset, ExperimentData
from hypex.dataset.roles import (
    Arg1Role,
    Arg2Role,
    InfoRole,
    TargetRole,
    FeatureRole,
    GroupingRole,
    PreTargetRole,
    StatisticRole,
    StratificationRole,
    TreatmentRole,
    TmpTreatmentRole,
    TempGroupingRole,
    TempTargetRole,
    ABCRole,
    default_roles,
)

__all__ = [
    "Dataset",
    "ExperimentData",
    "Arg1Role",
    "Arg2Role",
    "InfoRole",
    "TargetRole",
    "FeatureRole",
    "GroupingRole",
    "PreTargetRole",
    "StratificationRole",
    "StatisticRole",
    "TreatmentRole",
    "TmpTreatmentRole",
    "TempGroupingRole",
    "TempTargetRole",
    "ABCRole",
    "default_roles",
]
