from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Sequence

from ..dataset import ExperimentData
from ..dataset.ml_data import MLExperimentData
from ..executor import Executor
from ..transformers import TransformerMode
from ..utils.constants import DEFAULT_EXPERIMENT_DIR
from .artifact import ExperimentArtifact
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
        cleanup_after: Whether to cleanup ML artifacts after execution
        save_experiment: Save experiment artifact (transformers + models) for reuse
        load_experiment: Experiment ID to load fitted transformers and models from
        experiment_dir: Directory for saving/loading experiment artifacts (default: .hypex_experiments/)
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
    
    Example workflow with save/load:
        # Fit on virtual target
        ab_test = ABTest(
            enable_cupac=True,
            save_experiment=True
        )
        ab_test.execute(virtual_data)
        
        # Apply to real target
        ab_test = ABTest(
            enable_cupac=True,
            load_experiment="exp_id"
        )
        ab_test.execute(real_data)
    """
    
    def __init__(
        self,
        splitters: Sequence[Executor] | None = None,
        transformers: Sequence[Executor] | None = None,
        ml_executors: Sequence[Executor] | None = None,
        save_models: bool = False,
        cleanup_after: bool = True,
        save_experiment: bool = False,
        load_experiment: Optional[str] = None,
        experiment_dir: Optional[str] = None,
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
        self.cleanup_after = cleanup_after
        
        # Experiment artifact management
        self.save_experiment = save_experiment
        self.load_experiment = load_experiment
        self.experiment_dir = experiment_dir or self._get_default_experiment_dir()
        self._loaded_artifact: Optional[ExperimentArtifact] = None
    
    def execute(self, data: ExperimentData) -> ExperimentData:
        """
        Execute ML pipeline with automatic data transformation.
        
        Flow:
        1. Load artifact if load_experiment is specified
        2. Transform ExperimentData → MLExperimentData
        3. Configure transformers for inference mode if loading
        4. Execute splitters (prepare data structures)
        5. Execute transformers (preprocess data)
        6. Execute ml_executors (train/predict with ML models)
        7. Save artifact if save_experiment is True
        8. Transform back MLExperimentData → ExperimentData
        9. Optionally cleanup ML artifacts from memory
        """
        # Load artifact if specified
        if self.load_experiment is not None:
            self._load_artifact()
        
        # Transform to ML data
        ml_data = self._ensure_ml_data(data)
        
        # If loading, configure transformers for inference mode
        if self._loaded_artifact is not None:
            self._configure_for_inference(ml_data)
        
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
        
        # Save artifact if requested
        if self.save_experiment:
            self._save_artifact(experiment_data)
        
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
            trained_models = ml_data.ml.get('trained_models', {})
            model_stats = ml_data.ml.get('model_stats', {})
            
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
                f"Make sure you saved it with save_experiment=True"
            )
        
        self._loaded_artifact = ExperimentArtifact.load_from_directory(artifact_path)
        
        # Validate compatibility
        is_compatible, msg = self._loaded_artifact.validate_compatibility(self)
        if not is_compatible:
            raise ValueError(f"Incompatible experiment configuration: {msg}")
    
    def _configure_for_inference(self, ml_data: MLExperimentData) -> None:
        """
        Configure transformers for inference mode.
        Load fitted states from artifact into ml_data.
        Switch all transformers to TRANSFORM mode.
        Load trained models if available.
        """
        if self._loaded_artifact is None:
            return
        
        # Load transformer states from artifact
        transformer_states = self._loaded_artifact.load_transformer_states()
        
        # Add to ml_data
        for transformer_id, state in transformer_states.items():
            ml_data.add_fitted_transformer(transformer_id, state)
        
        # Switch all transformers to TRANSFORM mode
        for transformer in self.transformers:
            if hasattr(transformer, 'mode'):
                transformer.mode = TransformerMode.TRANSFORM
        
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
            ml_data.ml['trained_models'] = {}
            ml_data.ml['model_stats'] = {}
            
            for executor_id, executor_models in models_data.items():
                ml_data.ml['trained_models'][executor_id] = {}
                ml_data.ml['model_stats'][executor_id] = {}
                
                for target, model_info in executor_models.items():
                    ml_data.ml['trained_models'][executor_id][target] = model_info['model']
                    ml_data.ml['model_stats'][executor_id][target] = model_info['stats']
    
    @staticmethod
    def _get_default_experiment_dir() -> str:
        """Get default directory for experiment artifacts"""
        return os.path.join(os.getcwd(), DEFAULT_EXPERIMENT_DIR)
