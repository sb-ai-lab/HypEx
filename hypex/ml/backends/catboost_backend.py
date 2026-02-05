from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:
    from catboost import CatBoostRegressor

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostRegressor = None

from ...dataset import Dataset
from ...dataset.backends import PandasDataset
from .abstract import MLModelBackendBase


class CatBoostModelBackendBase(MLModelBackendBase):
    """
    Backend for CatBoost models.
    
    Поддерживает разные Dataset backends через dispatch.
    """
    
    BACKEND_MAPPING = {
        PandasDataset: "_fit_pandas",
    }
    
    def __init__(self, **kwargs):
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed")
        
        # Дефолтные параметры
        default_params = {"verbose": 0, "iterations": 100}
        default_params.update(kwargs)
        
        self._data = CatBoostRegressor(**default_params)
        self._is_fitted = False
        self._feature_names = None
    
    @property
    def data(self) -> Any:
        """Raw CatBoost model"""
        return self._data
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    @property
    def feature_names_(self) -> Optional[List[str]]:
        return self._feature_names
    
    def fit(self, X: Dataset, y: Dataset) -> CatBoostModelBackendBase:
        """Fit model - dispatches to backend-specific method"""
        backend_type = type(X.backend)
        
        if backend_type not in self.BACKEND_MAPPING:
            raise ValueError(f"Unsupported Dataset backend: {backend_type}")
        
        method_name = self.BACKEND_MAPPING[backend_type]
        method = getattr(self, method_name)
        
        self._feature_names = X.columns.tolist()
        return method(X, y)
    
    def predict(self, X: Dataset) -> np.ndarray:
        """Predict - dispatches to backend-specific method"""
        backend_type = type(X.backend)
        
        if backend_type not in self.BACKEND_MAPPING:
            raise ValueError(f"Unsupported Dataset backend: {backend_type}")
        
        method_name = self.BACKEND_MAPPING[backend_type].replace("_fit_", "_predict_")
        if not hasattr(self, method_name):
            method_name = "_predict_pandas"
        
        method = getattr(self, method_name)
        return method(X)
    
    def _fit_pandas(self, X: Dataset, y: Dataset) -> CatBoostModelBackendBase:
        """Fit using pandas backend"""
        X_data = X.backend.data
        y_data = y.backend.data
        
        # CatBoost can work directly with pandas
        y_series = y_data.iloc[:, 0] if hasattr(y_data, "iloc") else y_data
        
        self._data.fit(X_data, y_series)
        self._is_fitted = True
        return self
    
    def _predict_pandas(self, X: Dataset) -> np.ndarray:
        """Predict using pandas backend"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        
        X_data = X.backend.data
        return self._data.predict(X_data)
    
    def get_feature_importances(self) -> np.ndarray:
        """CatBoost native feature importances"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        return self._data.feature_importances_
    
    def clone(self) -> CatBoostModelBackendBase:
        """Clone unfitted model"""
        params = self._data.get_params()
        return CatBoostModelBackendBase(**params)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict"""
        return {
            "backend_type": "catboost",
            "model_params": self._data.get_params(),
            "is_fitted": self._is_fitted,
            "feature_names": self._feature_names,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CatBoostModelBackendBase:
        """Deserialize from dict"""
        backend = cls(**data["model_params"])
        backend._is_fitted = data["is_fitted"]
        backend._feature_names = data["feature_names"]
        return backend
    
    def save_model_file(self, path: str) -> None:
        """Save CatBoost model using native format"""
        if self._is_fitted:
            self._data.save_model(path)
    
    @classmethod
    def load_model_file(cls, path: str, metadata: Dict) -> CatBoostModelBackendBase:
        """Load CatBoost model from native format"""
        backend = cls(**metadata["model_params"])
        backend._data.load_model(path)
        backend._is_fitted = True
        backend._feature_names = metadata["feature_names"]
        return backend
