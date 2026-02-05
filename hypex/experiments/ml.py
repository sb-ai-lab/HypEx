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
    executes splitters → transformers → ml_executors in sequence,
    then transforms back → ExperimentData.
    
    Args:
        splitters: Data splitting executors (prepare train/test/val sets)
        transformers: Data transformation executors (normalize, encode, etc.)
        ml_executors: ML model training and prediction executors
        save_models: Whether to save trained models to disk
        models_dir: Directory for saving models
        load_models_dir: Directory for loading pre-trained models
        cleanup_after: Whether to cleanup ML artifacts after execution
        transformer: Whether to deepcopy data between executors
        key: Experiment identifier
    
    Example pipeline:
        Experiment(GroupSizes) → 
        MLExperiment(
            splitters=[CUPACDataSplitter()],
            transformers=[],
            ml_executors=[CUPACExecutor()]
        ) → 
        Experiment(GroupDifference, ABAnalyzer)
    """
    
    def __init__(
        self,
        splitters: Sequence[Executor] | None = None,
        transformers: Sequence[Executor] | None = None,
        ml_executors: Sequence[Executor] | None = None,
        save_models: bool = False,
        models_dir: Optional[str] = None,
        load_models_dir: Optional[str] = None,
        cleanup_after: bool = True,
        transformer: bool | None = None,
        key: Any = "",
    ):
        # Combine all executors for parent class
        all_executors = []
        if splitters:
            all_executors.extend(splitters)
        if transformers:
            all_executors.extend(transformers)
        if ml_executors:
            all_executors.extend(ml_executors)
        
        super().__init__(all_executors, transformer, key)
        
        self.splitters = list(splitters) if splitters else []
        self.transformers = list(transformers) if transformers else []
        self.ml_executors = list(ml_executors) if ml_executors else []
        self.save_models = save_models
        self.models_dir = models_dir
        self.load_models_dir = load_models_dir
        self.cleanup_after = cleanup_after
    
    def execute(self, data: ExperimentData) -> ExperimentData:
        """
        Execute ML pipeline with automatic data transformation.
        
        Flow:
        1. Transform ExperimentData → MLExperimentData
        2. Execute splitters (prepare data structures)
        3. Execute transformers (preprocess data)
        4. Execute ml_executors (train/predict with ML models)
        5. Transform back MLExperimentData → ExperimentData
        6. Optionally cleanup ML artifacts from memory
        """
        # Transform to ML data
        ml_data = self._ensure_ml_data(data)
        
        # Execute ML pipeline in order: splitters → transformers → ml_executors
        experiment_data = deepcopy(ml_data) if self.transformer else ml_data
        
        # Execute splitters
        for executor in self.splitters:
            executor.key = self.key
            experiment_data = executor.execute(experiment_data)
        
        # Execute transformers
        for executor in self.transformers:
            executor.key = self.key
            experiment_data = executor.execute(experiment_data)
        
        # Execute ML executors
        for executor in self.ml_executors:
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
