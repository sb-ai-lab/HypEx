from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from ..dataset import Dataset, ExperimentData
from ..utils import ID_SPLIT_SYMBOL, ExperimentDataEnum


class MLData:
    """
    Container for ML training/prediction data for a single target.
    
    This class encapsulates all data needed for ML model training:
    - Training data (X_train, Y_train)
    - Prediction data (X_predict) for current period
    - Cross-validation folds
    
    Attributes:
        X_train (Dataset): Training features
        Y_train (Dataset): Training target
        X_predict (Optional[Dataset]): Prediction features for current period
        crossval (Dict[int, Tuple[Dataset, Dataset]]): CV folds {fold_id: (X_val, Y_val)}
    """
    
    def __init__(
        self,
        X_train: Dataset,
        Y_train: Dataset,
        X_predict: Optional[Dataset] = None,
        crossval: Optional[Dict[int, Tuple[Dataset, Dataset]]] = None,
    ):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_predict = X_predict
        self.crossval = crossval or {}
    
    def __repr__(self) -> str:
        return (
            f"MLData(X_train={self.X_train.shape}, "
            f"Y_train={self.Y_train.shape}, "
            f"X_predict={self.X_predict.shape if self.X_predict else None}, "
            f"n_folds={len(self.crossval)})"
        )


class MLExperimentData(ExperimentData):
    """
    Extended ExperimentData with ML artifacts.
    
    Structure:
    - self.ml: Dict[str, MLData] - {target_name: MLData}
    - self.trained_models: Dict[str, Dict[str, MLModel]] - {executor_id: {target: MLModel}}
    - self.model_stats: Dict[str, Dict[str, ModelStats]] - {executor_id: {target: ModelStats}}
    - self.fitted_transformers: Dict[str, TransformerParams] - {transformer_id: TransformerParams}
    - self.config: Dict[str, Any] - configuration
    """
    
    def __init__(
        self,
        data: Dataset,
        save_models: bool = False,
    ):
        super().__init__(data=data)
        
        # Main ML data storage: {target_name: MLData}
        self.ml: Dict[str, MLData] = {}
        
        # Trained models and stats
        self.trained_models: Dict[str, Dict[str, MLModel]] = {}  # {executor_id: {target: MLModel}}
        self.model_stats: Dict[str, Dict[str, ModelStats]] = {}  # {executor_id: {target: ModelStats}}
        
        # Fitted transformers
        self.fitted_transformers: Dict[str, TransformerParams] = {}  # {transformer_id: TransformerParams}
        
        # Configuration
        self.config: Dict[str, Any] = {
            "save_models": save_models,
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
    
    # === MLData management ===
    
    def add_ml_data(self, target_name: str, ml_data: MLData) -> None:
        """
        Add MLData for a target.
        
        Args:
            target_name: Target column name
            ml_data: MLData instance with training/prediction data
        """
        self.ml[target_name] = ml_data
    
    def get_ml_data(self, target_name: str) -> MLData:
        """Get MLData for a target"""
        if target_name not in self.ml:
            raise KeyError(f"MLData not found for target '{target_name}'")
        return self.ml[target_name]
    
    def has_ml_data(self, target_name: str) -> bool:
        """Check if MLData exists for target"""
        return target_name in self.ml
    
    def get_all_targets(self) -> list[str]:
        """Get list of all targets with MLData"""
        return list(self.ml.keys())
    
    # === Trained models management ===
    
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
        
        Note:
            Models are stored in memory only. Disk saving happens through 
            ExperimentArtifact.save() when save_experiment=True.
        """
        # Store in trained_models
        if executor_id not in self.trained_models:
            self.trained_models[executor_id] = {}
            self.model_stats[executor_id] = {}
        
        self.trained_models[executor_id][target_name] = model
        self.model_stats[executor_id][target_name] = stats
        
        # Also store stats in analysis_tables for reporting
        stats_key = f"{executor_id}_{target_name}_stats"
        self.set_value(ExperimentDataEnum.analysis_tables, stats_key, stats.to_dict())
    
    def get_trained_model(self, executor_id: str, target_name: str) -> MLModel:
        """Get trained model by executor and target"""
        if executor_id in self.trained_models:
            if target_name in self.trained_models[executor_id]:
                return self.trained_models[executor_id][target_name]
        
        raise KeyError(
            f"Model not found: executor={executor_id}, target={target_name}"
        )
    
    def get_model_stats(self, executor_id: str, target_name: str) -> ModelStats:
        """Get model stats by executor and target"""
        if executor_id in self.model_stats:
            if target_name in self.model_stats[executor_id]:
                return self.model_stats[executor_id][target_name]
        raise KeyError(f"Stats not found: executor={executor_id}, target={target_name}")
    
    def get_all_models_for_target(self, target_name: str) -> Dict[str, MLModel]:
        """Get all models trained for a specific target across all executors"""
        result = {}
        for executor_id, models in self.trained_models.items():
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
        
        for executor_id, stats_dict in self.model_stats.items():
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
        self.trained_models.clear()
        # model_stats остаются для анализа
    
    # === Transformer state management ===
    
    def add_fitted_transformer(self, transformer_id: str, state: "TransformerParams") -> None:
        """
        Add fitted transformer state.
        
        Args:
            transformer_id: Transformer ID (e.g., "NaFiller__hash")
            state: TransformerParams with fitted parameters
        """
        self.fitted_transformers[transformer_id] = state
    
    def get_fitted_transformer(self, transformer_id: str) -> Optional["TransformerParams"]:
        """
        Get fitted transformer state.
        
        Returns:
            TransformerParams or None if not found
        """
        return self.fitted_transformers.get(transformer_id)
    
    def has_fitted_transformer(self, transformer_id: str) -> bool:
        """Check if transformer is fitted"""
        return transformer_id in self.fitted_transformers
    
    def get_all_fitted_transformers(self) -> Dict[str, "TransformerParams"]:
        """Get all fitted transformer states"""
        return self.fitted_transformers.copy()
    
    def cleanup_transformer_artifacts(self) -> None:
        """Clear all fitted transformer states"""
        self.fitted_transformers.clear()
