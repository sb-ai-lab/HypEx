from __future__ import annotations

import os
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence

from ..dataset import ExperimentData
from ..dataset.ml_data import MLExperimentData
from ..executor import Executor
from ..splitters.base import MLSplitter
from ..transformers import TransformerMode
from ..utils.constants import DEFAULT_EXPERIMENT_DIR
from .artifact import ExperimentArtifact
from .base import Experiment


class MLMode(str, Enum):
    """MLExperiment execution modes"""
    FIT_PREDICT = "fit_predict"  # Train and apply (default)
    FIT = "fit"                  # Only train, save models
    PREDICT = "predict"          # Only apply using saved models


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
        load_experiment: Experiment ID to load fitted transformers and models from (only for mode='predict')
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
            enable_cupac=True,
            cupac_mode='fit'
        )
        ab_test.execute(virtual_data)
        
        # Apply to real target
        ab_test = ABTest(
            enable_cupac=True,
            cupac_mode='predict',
            load_experiment="exp_id"
        )
        ab_test.execute(real_data)
    """
    
    def __init__(
        self,
        splitters: Sequence[MLSplitter] | None = None,
        transformers: Sequence[Executor] | None = None,
        ml_executors: Sequence[Executor] | None = None,
        mode: str | MLMode = MLMode.FIT_PREDICT,
        load_experiment: Optional[str] = None,
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
        if transformers:
            all_executors.extend(transformers)
        if ml_executors:
            all_executors.extend(ml_executors)
        
        super().__init__(all_executors, transformer, key)
        
        self.splitters = list(splitters) if splitters else []
        self.transformers = list(transformers) if transformers else []
        self.ml_executors = list(ml_executors) if ml_executors else []
        
        # Mode management
        self.mode = MLMode(mode) if isinstance(mode, str) else mode
        
        # Validate load_experiment usage
        if load_experiment is not None and self.mode != MLMode.PREDICT:
            raise ValueError(
                f"load_experiment can only be used with mode='predict', got mode='{self.mode.value}'"
            )
        
        if self.mode == MLMode.PREDICT and load_experiment is None:
            raise ValueError(
                "load_experiment is required when mode='predict'"
            )
        
        # Experiment artifact management
        self.load_experiment = load_experiment
        self.experiment_dir = self._get_default_experiment_dir()
        self._loaded_artifact: Optional[ExperimentArtifact] = None
        
        # Auto-determined properties based on mode
        self.save_models = self.mode in (MLMode.FIT, MLMode.FIT_PREDICT)
        self.save_experiment = self.mode in (MLMode.FIT, MLMode.FIT_PREDICT)
    
    def execute(self, data: ExperimentData) -> ExperimentData:
        """
        Execute ML pipeline with automatic data transformation.
        
        Flow:
        1. Load artifact if mode='predict'
        2. Transform ExperimentData → MLExperimentData
        3. Configure executors for correct mode
        4. Execute splitters (prepare data structures)
        5. Execute transformers (with mode mapping to fit/transform/fit_transform)
        6. Execute ml_executors (with mode mapping to fit/predict/fit_predict)
        7. Save artifact if mode='fit' or mode='fit_predict'
        8. Transform back MLExperimentData → ExperimentData
        9. Cleanup ML artifacts from memory
        """
        # Load artifact if in predict mode
        if self.mode == MLMode.PREDICT:
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
            experiment_data = executor.execute(experiment_data)
        
        # Execute transformers with mode mapping
        transformer_mode = self._map_mode_for_transformer()
        for executor in self.transformers:
            executor.key = self.key
            if hasattr(executor, 'mode'):
                executor.mode = transformer_mode
            experiment_data = executor.execute(experiment_data)
        
        # Execute ML executors with mode propagation
        for executor in self.ml_executors:
            executor.key = self.key
            if hasattr(executor, 'mode'):
                executor.mode = self.mode
            experiment_data = executor.execute(experiment_data)
        
        # Save artifact if in fit or fit_predict mode
        if self.save_experiment:
            self._save_artifact(experiment_data)
        
        # Transform back to regular ExperimentData
        result_data = experiment_data.to_experiment_data()
        
        # Always cleanup ML artifacts from memory
        experiment_data.cleanup_ml_artifacts()
        
        return result_data
    
    def _map_mode_for_transformer(self) -> TransformerMode:
        """
        Map MLExperiment mode to Transformer mode.
        
        MLExperiment.fit → Transformer.fit
        MLExperiment.predict → Transformer.transform
        MLExperiment.fit_predict → Transformer.fit_transform
        """
        if self.mode == MLMode.FIT:
            return TransformerMode.FIT
        elif self.mode == MLMode.PREDICT:
            return TransformerMode.TRANSFORM
        elif self.mode == MLMode.FIT_PREDICT:
            return TransformerMode.FIT_TRANSFORM
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
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
        
        # Save transformer states from ml_data
        artifact.transformer_states = ml_data.get_all_fitted_transformers()
        
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
        artifact_path = os.path.join(self.experiment_dir, self.load_experiment)
        
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
        
        # Load transformer states from artifact
        transformer_states = self._loaded_artifact.load_transformer_states()
        
        # Add to ml_data
        for transformer_id, state in transformer_states.items():
            ml_data.add_fitted_transformer(transformer_id, state)
        
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
