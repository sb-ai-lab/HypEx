from __future__ import annotations

from typing import Any, Literal

from .dataset import Dataset, ExperimentData
from ..transformers.abstract import Transformer
from ..utils import BackendsEnum


class MLExperimentData(ExperimentData):
    def __init__(
        self,
        data: ExperimentData,
        x_train: Dataset | None = None,
        y_train: Dataset | None = None,
        x_predict: Dataset | None = None,
        y_predict: Dataset | None = None,
        val: Dataset | None = None,
    ):
        # Initialize parent ExperimentData
        if isinstance(data, ExperimentData):
            super().__init__(data.ds)
            # Copy all fields from ExperimentData
            self.additional_fields = data.additional_fields
            self.variables = data.variables
            self.groups = data.groups
            self.analysis_tables = data.analysis_tables
            self.id_name_mapping = data.id_name_mapping
        else:
            super().__init__(data)
        
        # ML-specific fields
        self.x_train = x_train
        self.y_train = y_train
        self.x_predict = x_predict
        self.y_predict = y_predict
        self.val = val
        self._fitted_transformers: list[tuple[str, Transformer]] = []

    def to_experiment_data(self) -> ExperimentData:
        """Convert MLExperimentData back to ExperimentData.
        
        Returns:
            ExperimentData: Base experiment data without ML-specific fields.
        """
        experiment_data = ExperimentData(self.ds)
        experiment_data.additional_fields = self.additional_fields
        experiment_data.variables = self.variables
        experiment_data.groups = self.groups
        experiment_data.analysis_tables = self.analysis_tables
        experiment_data.id_name_mapping = self.id_name_mapping
        return experiment_data

    @staticmethod
    def from_experiment_data(
        experiment_data: ExperimentData,
        x_train: Dataset | None = None,
        y_train: Dataset | None = None,
        x_predict: Dataset | None = None,
        y_predict: Dataset | None = None,
        val: Dataset | None = None,
    ) -> MLExperimentData:
        """Create MLExperimentData from ExperimentData.
        
        Args:
            experiment_data: Base experiment data to convert.
            x_train: Training features dataset.
            y_train: Training target dataset.
            x_predict: Prediction features dataset.
            y_predict: Prediction target dataset (optional, for evaluation).
            val: Validation dataset.
        
        Returns:
            MLExperimentData: ML experiment data with additional fields.
        """
        return MLExperimentData(
            data=experiment_data,
            x_train=x_train,
            y_train=y_train,
            x_predict=x_predict,
            y_predict=y_predict,
            val=val,
        )
