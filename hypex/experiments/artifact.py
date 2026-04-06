"""Experiment artifact management - owns save/load logic."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from ..dataset.ml_data import MLExperimentData


class ExperimentArtifact:
    """
    Experiment artifact - manages saving/loading of fitted ML executor states and models.
    
    Owns all save/load logic for:
    - Metadata (experiment.json)
    - Fitted ML executor states (ml_executors/)
    - Trained models (models/)
    
    MLExperiment delegates to this class instead of doing it itself.
    
    Directory structure:
        experiment_20260210_123456/
            experiment.json          # Metadata + pipeline config
            ml_executors/            # Fitted ML executor states
                NaFiller_abc.json
                SomeExecutor_def.json
            models/                  # Trained models
                CUPACExecutor_123/
                    y/
                        sklearn_model.pkl
                        model_metadata.json
                        stats.json
    
    Examples:
        # Save
        artifact = ExperimentArtifact.create_from_experiment(ml_experiment, "exp_dir/")
        artifact.ml_executor_states = ml_data.get_all_fitted_ml_executors()
        artifact.save()
        
        # Load
        artifact = ExperimentArtifact.load_from_directory("exp_dir/")
        states = artifact.load_ml_executor_states()
    """
    
    def __init__(
        self,
        experiment_id: str,
        experiment_class: str,
        base_dir: str,
        pipeline_config: Dict[str, Any],
        created_at: Optional[str] = None,
    ):
        self.experiment_id = experiment_id
        self.experiment_class = experiment_class
        self.base_dir = base_dir
        self.pipeline_config = pipeline_config
        self.created_at = created_at or datetime.now().isoformat()
        self.ml_executor_states: Dict[str, Any] = {}  # Will be populated
        self.models_data: Dict[str, Any] = {}  # Will store trained models info
        
        # Paths
        self.metadata_file = os.path.join(base_dir, "experiment.json")
        self.ml_executors_dir = os.path.join(base_dir, "ml_executors")
        self.models_dir = os.path.join(base_dir, "models")
    
    # === Factory methods ===
    
    @classmethod
    def create_from_experiment(
        cls,
        ml_experiment: MLExperiment,
        base_dir: str,
        experiment_id: Optional[str] = None,
    ) -> ExperimentArtifact:
        """
        Create artifact from MLExperiment.
        
        Extracts pipeline config from executors.
        
        Args:
            ml_experiment: MLExperiment instance
            base_dir: Base directory for all experiments
            experiment_id: Optional experiment ID. If not provided, generates from key or timestamp
        """
        # Generate experiment ID if not provided
        if experiment_id is None:
            if ml_experiment.key:
                experiment_id = str(ml_experiment.key)
            else:
                experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Full path: base_dir / experiment_id
        full_path = os.path.join(base_dir, experiment_id)
        
        pipeline_config = {
            'splitters': [
                cls._serialize_executor(executor)
                for executor in ml_experiment.splitters
            ],
            'transformers': [
                cls._serialize_executor(executor)
                for executor in ml_experiment.transformers
            ],
            'ml_executors': [
                cls._serialize_executor(executor)
                for executor in ml_experiment.ml_executors
            ],
        }
        
        return cls(
            experiment_id=experiment_id,
            experiment_class=ml_experiment.__class__.__name__,
            base_dir=full_path,
            pipeline_config=pipeline_config,
        )
    
    @classmethod
    def load_from_directory(cls, base_dir: str) -> ExperimentArtifact:
        """Load artifact from directory"""
        metadata_file = os.path.join(base_dir, "experiment.json")
        
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(
                f"Experiment metadata not found: {metadata_file}"
            )
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        return cls(
            experiment_id=data["experiment_id"],
            experiment_class=data["experiment_class"],
            base_dir=base_dir,
            pipeline_config=data["pipeline_config"],
            created_at=data.get("created_at"),
        )
    
    @staticmethod
    def _serialize_executor(executor) -> Dict[str, Any]:
        """Serialize executor to dict"""
        config = {
            'class': executor.__class__.__name__,
            'key': executor.key,
            'id': executor.id,
        }
        
        # Save parameters
        if hasattr(executor, 'calc_kwargs'):
            config['params'] = executor.calc_kwargs
        
        # For CUPACExecutor - save cupac_models
        if hasattr(executor, 'cupac_models'):
            config['cupac_models'] = executor.cupac_models
        
        return config
    
    # === Save ===
    
    def save(self) -> None:
        """
        Save entire artifact (metadata + transformers + models).
        """
        # 1. Create directory structure
        self._create_directory_structure()
        
        # 2. Save metadata
        self._save_metadata()
        
        # 3. Save ML executor states
        self._save_ml_executor_states()
        
        # 4. Save models
        self._save_models()
    
    def _create_directory_structure(self) -> None:
        """Create directories for artifact"""
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.ml_executors_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def _save_metadata(self) -> None:
        """Save experiment.json"""
        metadata = {
            "experiment_id": self.experiment_id,
            "experiment_class": self.experiment_class,
            "created_at": self.created_at,
            "pipeline_config": self.pipeline_config,
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_ml_executor_states(self) -> None:
        """
        Save fitted ML executor states.

        Structure:
        ml_executors/
            StandardScaler_abc123.json  (JSON-serializable states)
            FaissMLExecutor_def456.pkl  (binary states via pickle)
        """
        import pickle

        for executor_id, state in self.ml_executor_states.items():
            if hasattr(state, 'to_dict'):
                state_dict = state.to_dict()
                # Try JSON serialization, fall back to pickle for binary objects
                try:
                    state_file = os.path.join(self.ml_executors_dir, f"{executor_id}.json")
                    with open(state_file, 'w') as f:
                        json.dump(state_dict, f, indent=2)
                except (TypeError, ValueError):
                    # Contains non-serializable objects, use pickle
                    state_file = os.path.join(self.ml_executors_dir, f"{executor_id}.pkl")
                    with open(state_file, 'wb') as f:
                        pickle.dump(state, f)
            else:
                # No to_dict method, use pickle directly
                state_file = os.path.join(self.ml_executors_dir, f"{executor_id}.pkl")
                with open(state_file, 'wb') as f:
                    pickle.dump(state, f)
    
    def _save_models(self) -> None:
        """
        Save trained models.
        
        Structure:
        models/
            CUPACExecutor_abc123/
                target_name/
                    sklearn_model.pkl (or catboost_model.cbm)
                    model_metadata.json
                    stats.json
        """
        if not self.models_data:
            return
        
        for executor_id, executor_models in self.models_data.items():
            executor_dir = os.path.join(self.models_dir, executor_id)
            os.makedirs(executor_dir, exist_ok=True)
            
            for target, model_info in executor_models.items():
                target_dir = os.path.join(executor_dir, target)
                os.makedirs(target_dir, exist_ok=True)
                
                # Save model using MLModel.save()
                model = model_info['model']
                model.save(target_dir)
                
                # Save stats
                stats = model_info['stats']
                stats_file = os.path.join(target_dir, "stats.json")
                with open(stats_file, 'w') as f:
                    json.dump(stats.to_dict(), f, indent=2)
    
    # === Load ===
    
    def load_ml_executor_states(self) -> Dict[str, Any]:
        """
        Load fitted ML executor states from disk.

        Returns:
            Dict of executor_id -> MLExecutorParams or pickled state
        """
        import pickle
        from ..executor.state import MLExecutorParams

        ml_executor_states = {}

        if not os.path.exists(self.ml_executors_dir):
            return ml_executor_states

        for filename in os.listdir(self.ml_executors_dir):
            state_file = os.path.join(self.ml_executors_dir, filename)

            if filename.endswith('.json'):
                executor_id = filename[:-5]
                with open(state_file, 'r') as f:
                    state_dict = json.load(f)
                state = MLExecutorParams.from_dict(state_dict)
                ml_executor_states[executor_id] = state

            elif filename.endswith('.pkl'):
                executor_id = filename[:-4]
                with open(state_file, 'rb') as f:
                    state = pickle.load(f)
                ml_executor_states[executor_id] = state

        return ml_executor_states
    
    def load_models(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Load trained models from disk.
        
        Returns:
            Dict structure:
            {
                'executor_id': {
                    'target_name': {
                        'model': MLModel,
                        'stats': ModelStats
                    }
                }
            }
        """
        from ..ml.models import MLModel
        from ..ml.stats import ModelStats
        
        models_data = {}
        
        if not os.path.exists(self.models_dir):
            return models_data
        
        # Iterate over executor directories
        for executor_id in os.listdir(self.models_dir):
            executor_dir = os.path.join(self.models_dir, executor_id)
            if not os.path.isdir(executor_dir):
                continue
            
            models_data[executor_id] = {}
            
            # Iterate over target directories
            for target in os.listdir(executor_dir):
                target_dir = os.path.join(executor_dir, target)
                if not os.path.isdir(target_dir):
                    continue
                
                # Load model
                model = MLModel.load(target_dir)
                
                # Load stats
                stats_file = os.path.join(target_dir, "stats.json")
                if os.path.exists(stats_file):
                    with open(stats_file, 'r') as f:
                        stats_dict = json.load(f)
                    stats = ModelStats.from_dict(stats_dict)
                else:
                    # Create minimal stats if not found
                    stats = ModelStats(
                        model_name=model._backend.__class__.__name__,
                        model_type=model._backend.__class__.__name__.replace("Backend", "").lower(),
                        feature_importances={},
                        training_time_seconds=0.0,
                    )
                
                models_data[executor_id][target] = {
                    'model': model,
                    'stats': stats
                }
        
        return models_data
    
    # === Validation ===
    
    def validate_compatibility(
        self,
        ml_experiment: MLExperiment
    ) -> tuple[bool, str]:
        """
        Check compatibility with current MLExperiment.
        
        Args:
            ml_experiment: Current MLExperiment
        
        Returns:
            (is_compatible, message)
        """
        saved_config = self.pipeline_config
        
        # Check transformer count
        saved_transformers = saved_config.get('transformers', [])
        current_transformers = ml_experiment.transformers
        
        if len(saved_transformers) != len(current_transformers):
            return False, (
                f"Transformer count mismatch: saved {len(saved_transformers)}, "
                f"current {len(current_transformers)}"
            )
        
        # Check transformer classes
        for idx, (saved, current) in enumerate(zip(saved_transformers, current_transformers)):
            if saved['class'] != current.__class__.__name__:
                return False, (
                    f"Transformer {idx}: expected {saved['class']}, "
                    f"got {current.__class__.__name__}"
                )
        
        return True, "Compatible"
    
    # === Utils ===
    
    def __repr__(self) -> str:
        return (
            f"ExperimentArtifact(id={self.experiment_id}, "
            f"class={self.experiment_class}, "
            f"dir={self.base_dir})"
        )
    
    def summary(self) -> str:
        """Text description of artifact"""
        lines = [
            f"Experiment: {self.experiment_id}",
            f"Class: {self.experiment_class}",
            f"Created: {self.created_at}",
            f"Directory: {self.base_dir}",
            "",
            "Pipeline:",
        ]
        
        for key in ['splitters', 'transformers', 'ml_executors']:
            items = self.pipeline_config.get(key, [])
            if items:
                lines.append(f"  {key.capitalize()}:")
                for item in items:
                    lines.append(f"    - {item['class']} (id={item['id']})")
        
        return "\n".join(lines)
