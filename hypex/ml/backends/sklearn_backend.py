from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from ...dataset import Dataset
from ...dataset.backends import PandasDataset
from .abstract import MLModelBackendBase


class SklearnModelBackendBase(MLModelBackendBase, ABC):
    """
    Base backend for sklearn models.
    
    Поддерживает разные Dataset backends через dispatch.
    """
    
    # Backend mapping для разных типов Dataset
    BACKEND_MAPPING = {
        PandasDataset: "_fit_pandas",
        # Можно добавить PolarsDataset: '_fit_polars' и т.д.
    }
    
    def __init__(self, model: Any):
        """
        Args:
            model: sklearn model instance
        """
        self._data = model  # raw sklearn model
        self._is_fitted = False
        self._feature_names = None
    
    @property
    def data(self) -> Any:
        """Raw sklearn model"""
        return self._data
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    @property
    def feature_names_(self) -> Optional[List[str]]:
        return self._feature_names
    
    def fit(self, X: Dataset, y: Dataset) -> SklearnModelBackendBase:
        """
        Fit model - dispatches to backend-specific method.
        
        Аналогично Dataset.__binary_magic_operator
        """
        backend_type = type(X.backend)
        
        if backend_type not in self.BACKEND_MAPPING:
            raise ValueError(f"Unsupported Dataset backend: {backend_type}")
        
        method_name = self.BACKEND_MAPPING[backend_type]
        method = getattr(self, method_name)
        
        self._feature_names = X.columns.tolist()
        return method(X, y)
    
    def predict(self, X: Dataset) -> Dataset:
        """Predict - dispatches to backend-specific method"""
        backend_type = type(X.backend)
        
        if backend_type not in self.BACKEND_MAPPING:
            raise ValueError(f"Unsupported Dataset backend: {backend_type}")
        
        method_name = self.BACKEND_MAPPING[backend_type].replace("_fit_", "_predict_")
        if not hasattr(self, method_name):
            method_name = "_predict_pandas"  # fallback
        
        method = getattr(self, method_name)
        return method(X)
    
    # Backend-specific implementations
    def _fit_pandas(self, X: Dataset, y: Dataset) -> SklearnModelBackendBase:
        """Fit using pandas backend"""
        X_data = X.backend.data  # pd.DataFrame
        y_data = y.backend.data  # pd.DataFrame or Series
        
        # Convert to numpy for sklearn
        X_np = X_data.values
        y_np = y_data.values.ravel() if hasattr(y_data, "values") else y_data
        
        self._data.fit(X_np, y_np)
        self._is_fitted = True
        return self
    
    def _predict_pandas(self, X: Dataset) -> Dataset:
        """Predict using pandas backend"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        
        X_data = X.backend.data
        X_np = X_data.values
        predictions = self._data.predict(X_np)
        
        # Return as Dataset
        return Dataset.from_dict(
            data={"prediction": predictions},
            roles={},
            index=X.index
        )
    
    def get_feature_importances(self) -> np.ndarray:
        """For linear models: absolute coefficients"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        return np.abs(self._data.coef_)
    
    def clone(self) -> SklearnModelBackendBase:
        """Clone unfitted model"""
        # Get model params and create new instance via the subclass __init__
        params = self._data.get_params()
        return self.__class__(**params)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict"""
        return {
            "backend_type": "sklearn",
            "model_class": self._data.__class__.__name__,
            "model_params": self._data.get_params(),
            "is_fitted": self._is_fitted,
            "feature_names": self._feature_names,
            "model_state": (
                {
                    "coef_": self._data.coef_.tolist() if self._is_fitted else None,
                    "intercept_": (
                        float(self._data.intercept_) if self._is_fitted else None
                    ),
                }
                if self._is_fitted
                else None
            ),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SklearnModelBackendBase:
        """Deserialize from dict"""
        # Recreate model
        model_class_name = data["model_class"]
        if model_class_name == "LinearRegression":
            model = LinearRegression(**data["model_params"])
        elif model_class_name == "Ridge":
            model = Ridge(**data["model_params"])
        elif model_class_name == "Lasso":
            model = Lasso(**data["model_params"])
        else:
            raise ValueError(f"Unknown sklearn model: {model_class_name}")
        
        backend = cls(model)
        
        # Restore fitted state
        if data["is_fitted"] and data["model_state"]:
            backend._data.coef_ = np.array(data["model_state"]["coef_"])
            backend._data.intercept_ = data["model_state"]["intercept_"]
            backend._is_fitted = True
        
        backend._feature_names = data["feature_names"]
        return backend
    
    def save_model_file(self, path: str) -> None:
        """Save sklearn model using joblib"""
        joblib.dump(self._data, path)
    
    @classmethod
    def load_model_file(cls, path: str, metadata: Dict) -> SklearnModelBackendBase:
        """Load sklearn model from joblib"""
        model = joblib.load(path)
        # Create backend by passing model params (not model itself) to match __init__ signature
        params = model.get_params()
        backend = cls(**params)
        # Then replace internal model with loaded one (already fitted)
        backend._data = model
        backend._is_fitted = metadata.get("is_fitted", True)
        backend._feature_names = metadata.get("feature_names")
        return backend


# Конкретные реализации для каждой модели
class LinearRegressionBackend(SklearnModelBackendBase):
    """Backend for Linear Regression"""
    
    def __init__(self, **kwargs):
        super().__init__(LinearRegression(**kwargs))


class RidgeBackend(SklearnModelBackendBase):
    """Backend for Ridge Regression"""
    
    def __init__(self, **kwargs):
        super().__init__(Ridge(**kwargs))


class LassoBackend(SklearnModelBackendBase):
    """Backend for Lasso Regression"""
    
    def __init__(self, **kwargs):
        super().__init__(Lasso(**kwargs))
