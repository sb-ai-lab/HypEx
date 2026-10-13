from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..dataset.dataset import Dataset
from ..dataset.roles import ABCRole, FeatureRole, TargetRole
from .experiment_data import ExperimentData

if TYPE_CHECKING:
    from ..executor import Executor



@dataclass
class MLData:
    X_role: ABCRole = FeatureRole
    Y_role: ABCRole = TargetRole

class MLExperimentData(ExperimentData):
    fitted_ml_executors: dict[str, Executor]
    ml_data: MLData

    def __init__(self, data):
        super().__init__(data)

    def to_experiment_data(self) -> ExperimentData:
        exp_data = ExperimentData(data=self.ds)
        exp_data.additional_fields = self.additional_fields
        exp_data.variables = self.variables
        exp_data.groups = self.groups
        exp_data.analysis_tables = self.analysis_tables
        exp_data.id_name_mapping = self.id_name_mapping

        return exp_data

    def cleanup_ml_artifacts() -> ExperimentData:
        ...

    def materialize(self) -> Dataset:
        ...
