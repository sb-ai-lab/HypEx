from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from ..dataset import Dataset, ExperimentData
from ..utils import ID_SPLIT_SYMBOL, ExperimentDataEnum

if TYPE_CHECKING:
    from ..ml.models import MLModel
    from ..ml.stats import ModelStats


class MLExperimentData(ExperimentData):
    """
    Extended ExperimentData with ML artifacts.
    
    Хранит ML данные в стандартной структуре HypEx:
    - self.ml['trained_models'][executor_id] = {target: MLModel}
    - self.ml['model_stats'][executor_id] = {target: ModelStats}
    - self.ml['config'] = {save_models: bool, models_dir: str}
    - self.ml['splits'] = prepared data from splitters
    """
    
    def __init__(
        self,
        data: Dataset,
        save_models: bool = False,
        models_dir: Optional[str] = None,
    ):
        super().__init__(data=data)
        
        # Инициализируем ml пространство
        self.ml: Dict[str, Any] = {
            "trained_models": {},  # {executor_id: {target: MLModel}}
            "model_stats": {},  # {executor_id: {target: ModelStats}}
            "config": {
                "save_models": save_models,
                "models_dir": models_dir or self._create_default_models_dir(),
            },
            "splits": None,  # Data prepared by splitters
        }
    
    @classmethod
    def from_experiment_data(
        cls, data: ExperimentData, save_models: bool = False
    ) -> MLExperimentData:
        """Transform ExperimentData → MLExperimentData"""
        ml_data = cls(
            data=data.ds,
            save_models=save_models
        )
        
        # Copy all standard attributes
        ml_data.additional_fields = data.additional_fields
        ml_data.variables = data.variables
        ml_data.groups = data.groups
        ml_data.analysis_tables = data.analysis_tables
        ml_data.id_name_mapping = data.id_name_mapping

        return ml_data
    
    def to_experiment_data(self) -> ExperimentData:
        """Transform MLExperimentData → ExperimentData (cleanup ML artifacts)"""
        exp_data = ExperimentData(data=self.ds)
        exp_data.additional_fields = self.additional_fields
        exp_data.variables = self.variables
        exp_data.groups = self.groups
        exp_data.analysis_tables = self.analysis_tables
        exp_data.id_name_mapping = self.id_name_mapping
        
        return exp_data
    
    def set_value(
        self,
        space: ExperimentDataEnum,
        executor_id: str | dict[str, str],
        value: Any,
        key: str | None = None,
        role=None,
    ) -> "MLExperimentData":
        """
        Override set_value to handle ML space.
        """
        if space == ExperimentDataEnum.ml:
            # For ml space, executor_id is the key (like "splits")
            # and value is the data to store
            self.ml[executor_id] = value
            return self
        else:
            # Delegate to parent for other spaces
            super().set_value(space, executor_id, value, key, role)
            return self
    
    def add_trained_model(
        self, executor_id: str, target_name: str, model: MLModel, stats: ModelStats
    ) -> None:
        """
        Add trained model with stats.
        
        Args:
            executor_id: ID of executor (e.g., "CUPACExecutor__<hash>")
            target_name: Target column name
            model: Trained MLModel
            stats: ModelStats for this model
        """
        # Store in ml space
        if executor_id not in self.ml["trained_models"]:
            self.ml["trained_models"][executor_id] = {}
            self.ml["model_stats"][executor_id] = {}
        
        self.ml["trained_models"][executor_id][target_name] = model
        self.ml["model_stats"][executor_id][target_name] = stats
        
        # Also store stats in analysis_tables for reporting
        stats_key = f"{executor_id}_{target_name}_stats"
        self.analysis_tables[stats_key] = stats.to_dict()
        
        # Save model if configured
        if self.ml["config"]["save_models"]:
            self._save_model_artifacts(executor_id, target_name, model, stats)
    
    def get_trained_model(self, executor_id: str, target_name: str) -> MLModel:
        """Get trained model by executor and target"""
        if executor_id in self.ml["trained_models"]:
            if target_name in self.ml["trained_models"][executor_id]:
                return self.ml["trained_models"][executor_id][target_name]
        
        # Try loading from disk if saved
        if self.ml["config"]["save_models"]:
            model_dir = self._get_model_path(executor_id, target_name)
            if os.path.exists(model_dir):
                return MLModel.load(model_dir)
        
        raise KeyError(
            f"Model not found: executor={executor_id}, target={target_name}"
        )
    
    def get_model_stats(self, executor_id: str, target_name: str) -> ModelStats:
        """Get model stats by executor and target"""
        if executor_id in self.ml["model_stats"]:
            if target_name in self.ml["model_stats"][executor_id]:
                return self.ml["model_stats"][executor_id][target_name]
        raise KeyError(f"Stats not found: executor={executor_id}, target={target_name}")
    
    def get_all_models_for_target(self, target_name: str) -> Dict[str, MLModel]:
        """Get all models trained for a specific target across all executors"""
        result = {}
        for executor_id, models in self.ml["trained_models"].items():
            if target_name in models:
                result[executor_id] = models[target_name]
        return result
    
    def get_best_model_for_target(
        self, target_name: str, metric: str = "variance_reduction_cv"
    ) -> Tuple[str, MLModel, ModelStats]:
        """
        Get best model for target based on metric.
        
        Returns:
            (executor_id, model, stats)
        """
        best_score = -float("inf")
        best_executor = None
        
        for executor_id, stats_dict in self.ml["model_stats"].items():
            if target_name in stats_dict:
                stats = stats_dict[target_name]
                score = getattr(stats, metric, 0.0)
                if score is not None and score > best_score:
                    best_score = score
                    best_executor = executor_id
        
        if best_executor is None:
            raise ValueError(f"No models found for target '{target_name}'")
        
        model = self.get_trained_model(best_executor, target_name)
        stats = self.get_model_stats(best_executor, target_name)
        return best_executor, model, stats
    
    def cleanup_ml_artifacts(self) -> None:
        """Free memory by clearing ML models (keep stats)"""
        self.ml["trained_models"].clear()
        # model_stats остаются для анализа
    
    def _save_model_artifacts(
        self, executor_id: str, target_name: str, model: MLModel, stats: ModelStats
    ) -> None:
        """Save model and stats to disk"""
        model_dir = self._get_model_path(executor_id, target_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model.save(model_dir)
        
        # Save stats separately
        stats_path = os.path.join(model_dir, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats.to_dict(), f, indent=2)
    
    def _get_model_path(self, executor_id: str, target_name: str) -> str:
        """Get directory path for model storage"""
        base_dir = self.ml["config"]["models_dir"]
        # Sanitize executor_id for filename
        safe_id = executor_id.replace(ID_SPLIT_SYMBOL, "_")
        return os.path.join(base_dir, safe_id, target_name)
    
    @staticmethod
    def _create_default_models_dir() -> str:
        """Create default directory for saving models in current working directory"""
        return os.getcwd()
