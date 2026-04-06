from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from ...dataset import Dataset


class MLModelBackendBase(ABC):
    """
    Abstract base for ML model backends (аналог DatasetBackendNavigation).
    
    Структура:
    - model_backend (SklearnModelBackendBase/CatBoostModelBackendBase)
      - model_backend.data (реальная sklearn/catboost модель)
    """
    
    @property
    def name(self) -> str:
        """Backend name (e.g., 'sklearn', 'catboost')"""
        return str(self.__class__.__name__).lower().replace("modelbackend", "")
    
    @property
    @abstractmethod
    def data(self) -> Any:
        """Raw model instance (sklearn/catboost object)"""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether model is fitted"""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def feature_names_(self) -> Optional[List[str]]:
        """Feature names used in training"""
        raise NotImplementedError
    
    # Core ML methods
    @abstractmethod
    def fit(self, X: Dataset, y: Dataset) -> MLModelBackendBase:
        """Fit model on Dataset (backend-specific implementation)"""
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, X: Dataset) -> Dataset:
        """Predict on Dataset (backend-specific implementation)"""
        raise NotImplementedError
    
    @abstractmethod
    def get_feature_importances(self) -> np.ndarray:
        """Get feature importances (backend-specific)"""
        raise NotImplementedError
    
    @abstractmethod
    def clone(self) -> MLModelBackendBase:
        """Clone unfitted model"""
        raise NotImplementedError
    
    # Serialization
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize backend state to dict"""
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> MLModelBackendBase:
        """Deserialize backend from dict"""
        raise NotImplementedError
    
    @abstractmethod
    def save_model_file(self, path: str) -> None:
        """Save model to file (backend-specific format)"""
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def load_model_file(cls, path: str, metadata: Dict) -> MLModelBackendBase:
        """Load model from file (backend-specific format)"""
        raise NotImplementedError
