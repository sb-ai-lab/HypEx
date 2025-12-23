from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator


@dataclass
class ModelCriteria:
    """Criteria for model evaluation and selection."""
    variance_reduction: float
    feature_importances: dict[str, float]
    model_name: str


class MLModel(ABC):
    """
    Abstract base class for machine learning models in HypEx.
    
    This class provides a unified interface for different ML libraries (sklearn, catboost, etc.)
    and handles model-specific feature importance extraction.
    
    Args:
        model: The underlying ML model instance.
        model_name: Name identifier for the model.
    """
    
    def __init__(self, model: BaseEstimator, model_name: str):
        self.model = model
        self.model_name = model_name
        self._is_fitted = False
    
    @abstractmethod
    def get_feature_importances(self, feature_names: list[str]) -> dict[str, float]:
        """
        Extract feature importances from the fitted model.
        
        Args:
            feature_names: List of feature names corresponding to model features.
            
        Returns:
            Dictionary mapping feature names to their importance values.
            
        Raises:
            ValueError: If model is not fitted.
        """
        raise NotImplementedError
    
    def fit(self, X: Any, y: Any) -> MLModel:
        """
        Fit the model to training data.
        
        Args:
            X: Training features.
            y: Training target.
            
        Returns:
            Self for method chaining.
        """
        self.model.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: Any) -> np.ndarray:
        """
        Generate predictions using the fitted model.
        
        Args:
            X: Features for prediction.
            
        Returns:
            Array of predictions.
            
        Raises:
            ValueError: If model is not fitted.
        """
        if not self._is_fitted:
            raise ValueError(f"Model {self.model_name} must be fitted before prediction")
        return self.model.predict(X)
    
    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._is_fitted
    
    def clone(self) -> MLModel:
        """Create a deep copy of the model."""
        from sklearn.base import clone
        return self.__class__(clone(self.model), self.model_name)


class SklearnLinearModel(MLModel):
    """
    Wrapper for sklearn linear models (LinearRegression, Ridge, Lasso).
    
    Feature importances are extracted from model coefficients.
    """
    
    def get_feature_importances(self, feature_names: list[str]) -> dict[str, float]:
        """Extract feature importances from linear model coefficients."""
        if not self._is_fitted:
            raise ValueError(f"Model {self.model_name} must be fitted before extracting importances")
        
        if not hasattr(self.model, 'coef_'):
            raise AttributeError(f"Model {self.model_name} does not have coef_ attribute")
        
        return {
            feature_name: float(self.model.coef_[i])
            for i, feature_name in enumerate(feature_names)
        }


class CatBoostModel(MLModel):
    """
    Wrapper for CatBoost models.
    
    Feature importances are extracted from the model's built-in feature_importances_.
    """
    
    def get_feature_importances(self, feature_names: list[str]) -> dict[str, float]:
        """Extract feature importances from CatBoost model."""
        if not self._is_fitted:
            raise ValueError(f"Model {self.model_name} must be fitted before extracting importances")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise AttributeError(f"Model {self.model_name} does not have feature_importances_ attribute")
        
        return {
            feature_name: float(self.model.feature_importances_[i])
            for i, feature_name in enumerate(feature_names)
        }


class MLModelRegistry:
    """
    Registry for managing available ML models.
    
    Provides centralized access to model instances and their configurations
    for different backends (pandas, polars, etc.).
    """
    
    def __init__(self):
        self._models: dict[str, dict[str, Optional[MLModel]]] = {}
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default models for CUPAC and other ML operations."""
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        
        # Register sklearn linear models
        self.register_model(
            'linear',
            pandasdataset=SklearnLinearModel(LinearRegression(), 'linear')
        )
        self.register_model(
            'ridge',
            pandasdataset=SklearnLinearModel(Ridge(), 'ridge')
        )
        self.register_model(
            'lasso',
            pandasdataset=SklearnLinearModel(Lasso(), 'lasso')
        )
        
        # Register CatBoost if available
        try:
            from catboost import CatBoostRegressor
            self.register_model(
                'catboost',
                pandasdataset=CatBoostModel(CatBoostRegressor(verbose=0), 'catboost')
            )
        except ImportError:
            pass
    
    def register_model(
        self,
        model_name: str,
        pandasdataset: Optional[MLModel] = None,
        polars: Optional[MLModel] = None
    ):
        """
        Register a model for different backends.
        
        Args:
            model_name: Unique identifier for the model.
            pandasdataset: MLModel instance for pandas backend.
            polars: MLModel instance for polars backend (currently unused).
        """
        self._models[model_name] = {
            'pandasdataset': pandasdataset,
            'polars': polars,
        }
    
    def get_model(self, model_name: str, backend: str = 'pandasdataset') -> Optional[MLModel]:
        """
        Retrieve a model instance for a specific backend.
        
        Args:
            model_name: Name of the model to retrieve.
            backend: Backend type ('pandasdataset' or 'polars').
            
        Returns:
            MLModel instance or None if not available.
            
        Raises:
            KeyError: If model_name is not registered.
        """
        if model_name not in self._models:
            raise KeyError(f"Model '{model_name}' not found in registry. Available models: {self.available_models}")
        return self._models[model_name].get(backend)
    
    @property
    def available_models(self) -> list[str]:
        """Get list of all registered model names."""
        return list(self._models.keys())
    
    def is_available(self, model_name: str, backend: str = 'pandasdataset') -> bool:
        """
        Check if a model is available for a specific backend.
        
        Args:
            model_name: Name of the model to check.
            backend: Backend type to check.
            
        Returns:
            True if model is available, False otherwise.
        """
        return (
            model_name in self._models
            and self._models[model_name].get(backend) is not None
        )


# Global model registry instance
MODEL_REGISTRY = MLModelRegistry()
