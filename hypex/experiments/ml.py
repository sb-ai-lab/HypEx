from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional, Sequence

from ..dataset import ExperimentData
from ..dataset.ml_data import MLExperimentData
from ..executor import Executor
from .base import Experiment


class MLExperiment(Experiment):
    """
    Experiment specialized for ML workflows with automatic data transformation.
    
    Transforms ExperimentData → MLExperimentData at start,
    executes ML executors, then transforms back → ExperimentData.
    
    Example pipeline:
        Experiment(GroupSizes) → 
        MLExperiment(Normalization, Cupac) → 
        Experiment(GroupDifference, ABAnalyzer)
    """
    
    def __init__(
        self,
        executors: Sequence[Executor],
        save_models: bool = False,
        models_dir: Optional[str] = None,
        load_models_dir: Optional[str] = None,
        cleanup_after: bool = True,
        transformer: bool | None = None,
        key: Any = "",
    ):
        super().__init__(executors, transformer, key)
        self.save_models = save_models
        self.models_dir = models_dir
        self.load_models_dir = load_models_dir
        self.cleanup_after = cleanup_after
    
    def execute(self, data: ExperimentData) -> ExperimentData:
        """
        Execute ML pipeline with automatic data transformation.
        
        Flow:
        1. Transform ExperimentData → MLExperimentData
        2. Execute all ML executors (they work with MLExperimentData)
        3. Transform back MLExperimentData → ExperimentData
        4. Optionally cleanup ML artifacts from memory
        """
        # Transform to ML data
        ml_data = self._ensure_ml_data(data)
        
        # Execute ML pipeline
        experiment_data = deepcopy(ml_data) if self.transformer else ml_data
        for executor in self.executors:
            executor.key = self.key
            experiment_data = executor.execute(experiment_data)
        
        # Transform back to regular ExperimentData
        result_data = experiment_data.to_experiment_data()
        
        # Cleanup if requested
        if self.cleanup_after:
            experiment_data.cleanup_ml_artifacts()
        
        return result_data
    
    def _ensure_ml_data(self, data: ExperimentData) -> MLExperimentData:
        """Convert to MLExperimentData if needed"""
        if isinstance(data, MLExperimentData):
            return data
        ml_data = MLExperimentData.from_experiment_data(
            data, save_models=self.save_models
        )
        # Override models_dir if specified
        if self.models_dir is not None:
            ml_data.ml["config"]["models_dir"] = self.models_dir
        # Set load_models_dir if specified
        if self.load_models_dir is not None:
            ml_data.ml["config"]["load_models_dir"] = self.load_models_dir
        return ml_data
