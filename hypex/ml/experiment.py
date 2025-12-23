from __future__ import annotations

from typing import Any, Sequence, Optional

from ..experiments.base import Experiment
from ..executor import Executor
from ..dataset.ml_data import MLExperimentData
from ..dataset import ExperimentData, Dataset


class MLExperiment(Experiment):
    """
    Specialized Experiment for ML workflows.
    
    This class extends the base Experiment with ML-specific functionality,
    ensuring proper handling of MLExperimentData and ML-specific execution patterns.
    
    Args:
        executors: Sequence of executors to run in the experiment.
        transformer: Whether this experiment transforms data.
        key: Unique identifier for the experiment.
        convert_to_ml_data: Whether to convert ExperimentData to MLExperimentData automatically.
    """
    
    def __init__(
        self,
        executors: Sequence[Executor],
        transformer: bool | None = None,
        key: Any = "",
        convert_to_ml_data: bool = True,
    ):
        super().__init__(executors, transformer, key)
        self.convert_to_ml_data = convert_to_ml_data
    
    def _ensure_ml_experiment_data(self, data: ExperimentData) -> MLExperimentData:
        """
        Ensure data is MLExperimentData instance.
        
        Args:
            data: Input ExperimentData.
            
        Returns:
            MLExperimentData instance.
        """
        if isinstance(data, MLExperimentData):
            return data
        
        if self.convert_to_ml_data:
            # Convert regular ExperimentData to MLExperimentData
            # Access ds property safely - all ExperimentData instances have it
            if not hasattr(data, 'ds'):
                raise TypeError(f"Expected ExperimentData instance, got {type(data)}")
            
            ml_data = MLExperimentData(data.ds)
            ml_data.additional_fields = data.additional_fields
            ml_data.variables = data.variables
            ml_data.groups = data.groups
            ml_data.analysis_tables = data.analysis_tables
            ml_data.id_name_mapping = data.id_name_mapping
            return ml_data
        
        return data
    
    def execute(self, data: ExperimentData) -> ExperimentData:
        """
        Execute the ML experiment.
        
        Args:
            data: Input experiment data.
            
        Returns:
            Processed experiment data (MLExperimentData if convert_to_ml_data is True).
        """
        # Convert to MLExperimentData if needed
        experiment_data = self._ensure_ml_experiment_data(data)
        
        # Execute parent logic
        return super().execute(experiment_data)
