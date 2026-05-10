from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Sequence

from ..dataset import ExperimentData
from ..dataset.ml_data import MLExperimentData
from ..executor import Executor
from ..executor.ml_executor import MLExecutor
from ..splitters.base import MLSplitter
from ..transformers import MLTransformer
from ..utils.constants import DEFAULT_EXPERIMENT_DIR
from ..utils.enums import MLModeEnum
from .artifact import ExperimentArtifact
from .base import Experiment


class MLExperiment(Experiment):
    """
    Experiment specialized for ML workflows with automatic data transformation.
    
    Transforms ExperimentData → MLExperimentData at start,
    executes splitters → transformers → ml_executors in sequence,
    then transforms back → ExperimentData.
    
    Supports three execution modes:
    - fit_predict: Train models and apply them (default behavior)
    - fit: Only train models, save for later use
    - predict: Only apply pre-trained models
    
    Args:
        splitters: Data splitting executors (prepare train/test/val sets)
        transformers: Data transformation executors (normalize, encode, etc.)
        ml_executors: ML model training and prediction executors
        mode: Execution mode - 'fit', 'predict', or 'fit_predict' (default: 'fit_predict')
        experiment_id: Experiment ID to load fitted transformers and models from (only for mode='predict')
        transformer: Whether to deepcopy data between executors
        key: Experiment identifier
    
    Example pipeline:
        Experiment(GroupSizes) → 
        MLExperiment(
            splitters=[CUPACDataSplitter()],
            transformers=[],
            ml_executors=[CUPACExecutor()],
            mode='fit_predict'
        ) → 
        Experiment(GroupDifference, ABAnalyzer)
    
    Example workflow with fit/predict:
        # Fit on virtual target
        ab_test = ABTest(
            cupac_mode='fit'
        )
        ab_test.execute(virtual_data)
        
        # Apply to real target
        ab_test = ABTest(
            cupac_mode='predict',
            experiment_id="exp_id"
        )
        ab_test.execute(real_data)
    """
    
    def __init__(
        self,
        splitters: Sequence[MLSplitter] | None = None,
        transformers: Sequence[MLTransformer] | None = None,
        ml_executors: Sequence[MLExecutor] | None = None,
        mode: str | MLModeEnum = MLModeEnum.FIT_PREDICT,
        experiment_id: Optional[str] = None,
        transformer: bool | None = None,
        key: Any = "",
    ):
        # Combine all executors for parent class
        all_executors = []
        if splitters and not all(isinstance(splitter, MLSplitter) for splitter in splitters):
            wrong = [type(splitter).__name__ for splitter in splitters if not isinstance(splitter, MLSplitter)]
            raise TypeError(
                "MLExperiment.splitters must contain only MLSplitter instances. "
                f"Got: {wrong}"
            )
        if splitters:
            all_executors.extend(splitters)
        if transformers and not all(isinstance(transformer, MLTransformer) for transformer in transformers):
            wrong = [
                type(transformer).__name__
                for transformer in transformers
                if not isinstance(transformer, MLTransformer)
            ]
            raise TypeError(
                "MLExperiment.transformers must contain only MLTransformer instances. "
                f"Got: {wrong}"
            )
        if transformers:
            all_executors.extend(transformers)
        if ml_executors and not all(isinstance(executor, MLExecutor) for executor in ml_executors):
            wrong = [type(executor).__name__ for executor in ml_executors if not isinstance(executor, MLExecutor)]
            raise TypeError(
                "MLExperiment.ml_executors must contain only MLExecutor instances. "
                f"Got: {wrong}"
            )
        if ml_executors:
            all_executors.extend(ml_executors)
        
        super().__init__(all_executors, transformer, key)
        
        self.splitters = list(splitters) if splitters else []
        self.transformers = list(transformers) if transformers else []
        self.ml_executors = list(ml_executors) if ml_executors else []
        
        # Mode management
        self.mode = MLModeEnum(mode) if isinstance(mode, str) else mode
        
        # Validate experiment_id usage
        if experiment_id is not None and self.mode != MLModeEnum.PREDICT:
            raise ValueError(
                f"experiment_id can only be used with mode='predict', got mode='{self.mode.value}'"
            )
        
        if self.mode == MLModeEnum.PREDICT and experiment_id is None:
            raise ValueError(
                "experiment_id is required when mode='predict'"
            )
        
        # Experiment artifact management
        self.experiment_id = experiment_id
        self.experiment_dir = self._get_default_experiment_dir()
        self._loaded_artifact: Optional[ExperimentArtifact] = None
        
        # Auto-determined properties based on mode
        self.save_models = self.mode in (MLModeEnum.FIT, MLModeEnum.FIT_PREDICT)
        self.save_experiment = self.mode in (MLModeEnum.FIT, MLModeEnum.FIT_PREDICT)
    
    def execute(self, data: ExperimentData) -> ExperimentData:
        """
        Execute ML pipeline with automatic data transformation.
        
        Flow:
        1. Load artifact if mode='predict'
        2. Transform ExperimentData → MLExperimentData
        3. Configure executors for correct mode
        4. Execute splitters (prepare data structures)
        5. Execute ML transformers (with MLModeEnum propagation)
        6. Execute ml_executors (with MLModeEnum propagation)
        7. Save artifact if mode='fit' or mode='fit_predict'
        8. Transform back MLExperimentData → ExperimentData
        9. Cleanup ML artifacts from memory
        """
        # Load artifact if in predict mode
        if self.mode == MLModeEnum.PREDICT:
            self._load_artifact()
        
        # Transform to ML data
        ml_data = self._ensure_ml_data(data)
        
        # Configure for inference if loading
        if self._loaded_artifact is not None:
            self._configure_for_inference(ml_data)
        
        # Execute ML pipeline in order: splitters → transformers → ml_executors
        experiment_data = deepcopy(ml_data) if self.transformer else ml_data
        
        # Execute splitters (always same behavior)
        for executor in self.splitters:
            executor.key = self.key
            experiment_data = executor.execute(experiment_data, mode=self.mode)
        
        # Execute ML transformers with mode propagation
        for executor in self.transformers:
            executor.key = self.key
            experiment_data = executor.execute(experiment_data, mode=self.mode)
        
        # Execute ML executors with mode propagation
        for executor in self.ml_executors:
            executor.key = self.key
            experiment_data = executor.execute(experiment_data, mode=self.mode)
        
        # Save artifact if in fit or fit_predict mode
        if self.save_experiment:
            self._save_artifact(experiment_data)
        
        # Transform back to regular ExperimentData
        result_data = experiment_data.to_experiment_data()
        
        # Always cleanup ML artifacts from memory
        experiment_data.cleanup_ml_artifacts()
        experiment_data.cleanup_ml_executor_artifacts()
        
        return result_data
    
    def _ensure_ml_data(self, data: ExperimentData) -> MLExperimentData:
        """Convert to MLExperimentData if needed"""
        if isinstance(data, MLExperimentData):
            return data
        ml_data = MLExperimentData.from_experiment_data(
            data, save_models=self.save_models
        )
        return ml_data
    
    def _save_artifact(self, ml_data: MLExperimentData) -> None:
        """Save experiment artifact with models and transformer states"""
        artifact = ExperimentArtifact.create_from_experiment(
            ml_experiment=self,
            base_dir=self.experiment_dir,
            experiment_id=self.key if self.key else None
        )
        
        # Save fitted ML executor states from ml_data
        artifact.ml_executor_states = ml_data.get_all_fitted_ml_executors()
        
        # Save trained models from ml_data (combine models + stats)
        if self.save_models:
            trained_models = ml_data.trained_models
            model_stats = ml_data.model_stats
            
            # Combine into {executor_id: {target: {'model': MLModel, 'stats': ModelStats}}}
            artifact.models_data = {}
            for executor_id, executor_models in trained_models.items():
                artifact.models_data[executor_id] = {}
                for target, model in executor_models.items():
                    stats = model_stats.get(executor_id, {}).get(target)
                    artifact.models_data[executor_id][target] = {
                        'model': model,
                        'stats': stats
                    }
        
        artifact.save()
    
    def _load_artifact(self) -> None:
        """Load experiment artifact"""
        # User passes experiment ID, construct full path
        artifact_path = os.path.join(self.experiment_dir, self.experiment_id)
        
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"Experiment artifact not found at {artifact_path}. "
                f"Make sure you saved it with mode='fit' or mode='fit_predict'"
            )
        
        self._loaded_artifact = ExperimentArtifact.load_from_directory(artifact_path)
        
        # Validate compatibility
        is_compatible, msg = self._loaded_artifact.validate_compatibility(self)
        if not is_compatible:
            raise ValueError(f"Incompatible experiment configuration: {msg}")
    
    def _configure_for_inference(self, ml_data: MLExperimentData) -> None:
        """
        Configure for inference mode (predict).
        Load fitted states from artifact into ml_data.
        Load trained models if available.
        """
        if self._loaded_artifact is None:
            return
        
        # Load fitted ML executor states from artifact
        ml_executor_states = self._loaded_artifact.load_ml_executor_states()

        # Add to ml_data
        for executor_id, state in ml_executor_states.items():
            ml_data.add_fitted_ml_executor(executor_id, state)
        
        # Restore executor IDs from saved configuration
        # This ensures that loaded models can be found by their original executor IDs
        saved_config = self._loaded_artifact.pipeline_config
        
        # Restore ML executor IDs
        for i, executor in enumerate(self.ml_executors):
            if i < len(saved_config.get('ml_executors', [])):
                saved_executor = saved_config['ml_executors'][i]
                executor._id = saved_executor['id']
        
        # Restore transformer IDs (if any)
        for i, transformer in enumerate(self.transformers):
            if i < len(saved_config.get('transformers', [])):
                saved_transformer = saved_config['transformers'][i]
                transformer._id = saved_transformer['id']
        
        # Load trained models if available
        models_data = self._loaded_artifact.load_models()
        if models_data:
            # Split into trained_models and model_stats
            ml_data.trained_models = {}
            ml_data.model_stats = {}
            
            for executor_id, executor_models in models_data.items():
                ml_data.trained_models[executor_id] = {}
                ml_data.model_stats[executor_id] = {}
                
                for target, model_info in executor_models.items():
                    ml_data.trained_models[executor_id][target] = model_info['model']
                    ml_data.model_stats[executor_id][target] = model_info['stats']
    
    @staticmethod
    def _get_default_experiment_dir() -> str:
        """Get default directory for experiment artifacts"""
        return os.path.join(os.getcwd(), DEFAULT_EXPERIMENT_DIR)
