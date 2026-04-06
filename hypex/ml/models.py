from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
from sklearn.model_selection import KFold

from ..dataset import Dataset
from .backends.abstract import MLModelBackendBase
from .backends.catboost_backend import CatBoostModelBackendBase
from .backends.sklearn_backend import LassoBackend, LinearRegressionBackend, RidgeBackend
from .stats import ModelStats


class MLModel:
    """
    High-level ML model wrapper (аналог Dataset).
    
    Структура:
    MLModel
      └─ model_backend (LinearRegressionBackend/RidgeBackend/CatBoostModelBackendBase)
           └─ model_backend.data (sklearn LinearRegression/CatBoostRegressor)
    
    MLModel работает с Dataset, backend обрабатывает специфику данных.
    """
    
    # Registry для создания моделей
    _MODEL_BACKENDS = {
        "linear": LinearRegressionBackend,
        "ridge": RidgeBackend,
        "lasso": LassoBackend,
        "catboost": CatBoostModelBackendBase,
    }
    
    @classmethod
    def create(cls, model_type: str, **kwargs) -> MLModel:
        """
        Create MLModel with appropriate backend.
        
        Args:
            model_type: 'linear', 'ridge', 'lasso', 'catboost'
            **kwargs: hyperparameters passed to backend
        
        Returns:
            MLModel instance
        
        Example:
            >>> model = MLModel.create('linear')
            >>> model = MLModel.create('ridge', alpha=1.0)
            >>> model = MLModel.create('catboost', iterations=200)
        """
        if model_type not in cls._MODEL_BACKENDS:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(cls._MODEL_BACKENDS.keys())}"
            )
        
        backend_class = cls._MODEL_BACKENDS[model_type]
        backend = backend_class(**kwargs)
        
        return cls(backend, model_type, **kwargs)
    
    def __init__(
        self, model_backend: MLModelBackendBase, model_type: str, **metadata
    ):
        """
        Args:
            model_backend: Backend instance (SklearnModelBackendBase/CatBoostModelBackendBase)
            model_type: Model type string
            **metadata: Additional metadata
        """
        self._backend = model_backend
        self.model_type = model_type
        self.metadata = metadata
        self.metadata["created_at"] = datetime.now().isoformat()
    
    @property
    def backend(self) -> MLModelBackendBase:
        """Model backend (аналог Dataset.backend)"""
        return self._backend
    
    @property
    def feature_names(self) -> List[str]:
        """Feature names used in training"""
        return self._backend.feature_names_ or []
    
    @property
    def is_fitted(self) -> bool:
        """Whether model is fitted"""
        return self._backend.is_fitted
    
    def fit(self, X: Dataset, y: Dataset) -> MLModel:
        """
        Fit model on Dataset.
        
        Backend автоматически определяет тип Dataset и использует нужный метод.
        
        Args:
            X: Features Dataset (может быть PandasDataset, PolarsDataset, etc.)
            y: Target Dataset
        
        Returns:
            self (fitted model)
        """
        self._backend.fit(X, y)
        return self
    
    def predict(self, X: Dataset) -> Dataset:
        """
        Predict on Dataset.
        
        Backend автоматически определяет тип Dataset.
        
        Args:
            X: Features Dataset
        
        Returns:
            Predictions as Dataset
        """
        return self._backend.predict(X)
    
    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importances as dict.
        
        Returns:
            Dict {feature_name: importance}
        """
        importances = self._backend.get_feature_importances()
        return dict(zip(self.feature_names, importances))
    
    def clone(self) -> MLModel:
        """Clone unfitted model"""
        new_backend = self._backend.clone()
        return MLModel(new_backend, self.model_type, **self.metadata)
    
    def cross_validate(
        self,
        X: Dataset,
        y: Dataset,
        n_folds: int = 5,
        random_state: Optional[int] = None,
        aggregation: str = "mean",
        metric_func: Optional[Callable] = None,
    ) -> ModelStats:
        """
        Perform cross-validation and return aggregated ModelStats.
        
        Args:
            X: Features Dataset
            y: Target Dataset
            n_folds: Number of CV folds
            random_state: Random seed
            aggregation: 'mean', 'median', 'max' for aggregating fold stats
            metric_func: Custom metric function (y_true, y_pred) -> score
                        Default: variance_reduction
        
        Returns:
            ModelStats with aggregated CV results
        """
        # Note: sklearn KFold requires numpy arrays for indexing,
        # so we temporarily access backend.data here for technical reasons.
        # This is an exception - the rest of the code uses Dataset API.
        X_data = X.backend.data
        y_data = y.backend.data
        
        # Convert to numpy for splitting
        if hasattr(X_data, "values"):
            X_np = X_data.values
        else:
            X_np = np.array(X_data)
        
        if hasattr(y_data, "values"):
            y_np = (
                y_data.values.ravel()
                if hasattr(y_data.values, "ravel")
                else y_data.values
            )
        else:
            y_np = np.array(y_data).ravel()
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        fold_scores = []
        fold_importances = []
        
        start_time = time.time()
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_np)):
            # Create fold datasets
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Clone and fit model
            fold_model = self.clone()
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Predict (returns Dataset)
            y_pred_ds = fold_model.predict(X_val_fold)
            y_pred = np.array(y_pred_ds.get_values(column="prediction"))
            
            # Calculate score
            if metric_func is None:
                # Default: variance reduction for CUPAC
                y_val_col = y_val_fold.columns[0]
                y_val_np = np.array(y_val_fold.get_values(column=y_val_col))
                
                # mean() returns float for single column Dataset
                y_train_mean = float(y_train_fold.mean())
                
                y_adjusted = y_val_np - y_pred + y_train_mean
                score = self._calculate_variance_reduction(y_val_np, y_adjusted)
            else:
                y_val_col = y_val_fold.columns[0]
                y_val_np = np.array(y_val_fold.get_values(column=y_val_col))
                score = metric_func(y_val_np, y_pred)
            
            fold_scores.append(score)
            fold_importances.append(fold_model.get_feature_importances())
        
        training_time = time.time() - start_time
        
        # Aggregate results
        aggregated_importances = self._aggregate_importances(
            fold_importances, aggregation
        )
        aggregated_score = self._aggregate_scores(fold_scores, aggregation)
        
        # Create ModelStats
        return ModelStats(
            model_name=self.model_type,
            model_type=self.model_type,
            cv_score=aggregated_score,
            cv_scores_per_fold=fold_scores,
            feature_importances=aggregated_importances,
            n_folds=n_folds,
            training_time_seconds=training_time,
            aggregation_method=aggregation,
        )
    
    @staticmethod
    def _calculate_variance_reduction(
        y_original: np.ndarray, y_adjusted: np.ndarray
    ) -> float:
        """Calculate variance reduction percentage"""
        var_orig = np.var(y_original)
        var_adj = np.var(y_adjusted)
        if var_orig < 1e-10:
            return 0.0
        return float(max(0, (1 - var_adj / var_orig) * 100))
    
    @staticmethod
    def _aggregate_importances(
        fold_importances: List[Dict[str, float]], method: str
    ) -> Dict[str, float]:
        """Aggregate feature importances across folds"""
        if not fold_importances:
            return {}
        
        features = fold_importances[0].keys()
        
        if method == "mean":
            agg_func = np.mean
        elif method == "median":
            agg_func = np.median
        elif method == "max":
            agg_func = np.max
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        return {
            feature: float(agg_func([fold[feature] for fold in fold_importances]))
            for feature in features
        }
    
    @staticmethod
    def _aggregate_scores(scores: List[float], method: str) -> float:
        """Aggregate CV scores"""
        if method == "mean":
            return float(np.mean(scores))
        elif method == "median":
            return float(np.median(scores))
        elif method == "max":
            return float(np.max(scores))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def save(self, directory: str) -> Dict[str, str]:
        """
        Save model and metadata to directory.
        
        Returns:
            Dict with saved file paths
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save backend model file
        if self.model_type == "catboost":
            model_filename = "catboost_model.cbm"
        else:
            model_filename = "sklearn_model.pkl"
        
        model_path = os.path.join(directory, model_filename)
        self._backend.save_model_file(model_path)
        
        # Save metadata
        metadata = {
            "model_type": self.model_type,
            "backend_data": self._backend.to_dict(),
            "model_filename": model_filename,
            "metadata": self.metadata,
        }
        
        meta_path = os.path.join(directory, "model_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return {"model": model_path, "metadata": meta_path}
    
    @classmethod
    def load(cls, directory: str) -> MLModel:
        """Load model from directory"""
        # Load metadata
        meta_path = os.path.join(directory, "model_metadata.json")
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        
        model_type = metadata["model_type"]
        backend_data = metadata["backend_data"]
        model_filename = metadata["model_filename"]
        
        # Load backend
        backend_class = cls._MODEL_BACKENDS[model_type]
        model_path = os.path.join(directory, model_filename)
        backend = backend_class.load_model_file(model_path, backend_data)
        
        return cls(backend, model_type, **metadata["metadata"])
