from __future__ import annotations

from typing import Any, Optional

from ..dataset.dataset import Dataset, ExperimentData
from .stats import MLExecutionStats, ModelStats


class MLExperimentData(ExperimentData):
    """
    Extended ExperimentData for ML-specific workflows.
    
    This class extends the base ExperimentData with ML-specific functionality
    for storing model statistics, trained models, and ML execution metadata.
    
    Attributes:
        ml_stats: Statistics from ML model execution.
        trained_models: Dictionary of trained models by target and model name.
        ml_metadata: Additional ML-specific metadata.
    """
    
    def __init__(self, data: Dataset):
        super().__init__(data)
        self.ml_stats: Optional[MLExecutionStats] = None
        self.trained_models: dict[str, Any] = {}
        self.ml_metadata: dict[str, Any] = {}
    
    def set_ml_stats(self, stats: MLExecutionStats) -> MLExperimentData:
        """
        Set ML execution statistics.
        
        Args:
            stats: MLExecutionStats instance with execution results.
            
        Returns:
            Self for method chaining.
        """
        self.ml_stats = stats
        return self
    
    def get_model_stats(self, target: str) -> Optional[ModelStats]:
        """
        Get model statistics for a specific target.
        
        Args:
            target: Target name to retrieve statistics for.
            
        Returns:
            ModelStats for the target, or None if not found.
        """
        if self.ml_stats is None:
            return None
        return self.ml_stats.target_stats.get(target)
    
    def store_trained_model(
        self, 
        target: str, 
        model_name: str, 
        model: Any
    ) -> MLExperimentData:
        """
        Store a trained model for later use.
        
        Args:
            target: Target name the model was trained for.
            model_name: Name of the model.
            model: Trained model instance.
            
        Returns:
            Self for method chaining.
        """
        key = f"{target}_{model_name}"
        self.trained_models[key] = model
        return self
    
    def get_trained_model(self, target: str, model_name: str) -> Optional[Any]:
        """
        Retrieve a stored trained model.
        
        Args:
            target: Target name.
            model_name: Model name.
            
        Returns:
            Trained model instance, or None if not found.
        """
        key = f"{target}_{model_name}"
        return self.trained_models.get(key)
    
    def set_ml_metadata(self, key: str, value: Any) -> MLExperimentData:
        """
        Store ML-specific metadata.
        
        Args:
            key: Metadata key.
            value: Metadata value.
            
        Returns:
            Self for method chaining.
        """
        self.ml_metadata[key] = value
        return self
    
    def get_ml_metadata(self, key: str, default: Any = None) -> Any:
        """
        Retrieve ML-specific metadata.
        
        Args:
            key: Metadata key.
            default: Default value if key not found.
            
        Returns:
            Metadata value or default.
        """
        return self.ml_metadata.get(key, default)
    
    def copy(self, data: Dataset | None = None) -> MLExperimentData:
        """
        Create a deep copy of the MLExperimentData.
        
        Args:
            data: Optional new dataset to use in the copy.
            
        Returns:
            New MLExperimentData instance with copied data.
        """
        from copy import deepcopy
        result = MLExperimentData(data if data is not None else deepcopy(self._data))
        
        # Copy all parent class attributes
        result.additional_fields = deepcopy(self.additional_fields)
        result.variables = deepcopy(self.variables)
        result.groups = deepcopy(self.groups)
        result.analysis_tables = deepcopy(self.analysis_tables)
        result.id_name_mapping = deepcopy(self.id_name_mapping)
        
        # Copy ML-specific attributes
        result.ml_stats = deepcopy(self.ml_stats)
        result.trained_models = deepcopy(self.trained_models)
        result.ml_metadata = deepcopy(self.ml_metadata)
        
        return result
