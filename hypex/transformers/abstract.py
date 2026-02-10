from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, TYPE_CHECKING

from ..dataset import Dataset, ExperimentData
from ..executor import Calculator
from ..utils import AbstractMethodError
from .state import TransformerState

if TYPE_CHECKING:
    from ..dataset.ml_data import MLExperimentData


class TransformerMode(str, Enum):
    """Transformer execution modes"""
    FIT_TRANSFORM = "fit_transform"  # Fit and apply (default, backwards compatible)
    FIT = "fit"                      # Only fit, save state
    TRANSFORM = "transform"          # Only apply using saved state


class Transformer(Calculator):
    """
    Base class for transformers with fit/transform support.
    
    Supports three modes:
    1. fit_transform - fit and apply (default, backwards compatible)
    2. fit - only fit, save state to MLExperimentData
    3. transform - apply using saved state from MLExperimentData
    
    For stateless transformers (old style), no changes needed.
    For stateful transformers, override _fit() and _transform().
    
    Examples:
        # Mode 1: Fit + Transform (default)
        transformer = NaFiller(method="mean")
        result = transformer.execute(data)  # fits and applies
        
        # Mode 2: Only Fit
        transformer = NaFiller(method="mean", mode="fit")
        result = transformer.execute(data)  # only fits, saves state
        
        # Mode 3: Only Transform
        transformer = NaFiller(method="mean", mode="transform")
        result = transformer.execute(data)  # uses saved state
    """
    
    def __init__(
        self,
        mode: str | TransformerMode = TransformerMode.FIT_TRANSFORM,
        key: Any = "",
        **kwargs
    ):
        super().__init__(key=key, **kwargs)
        self.mode = TransformerMode(mode) if isinstance(mode, str) else mode
        self._fitted_state: Optional[TransformerState] = None
    
    @property
    def _is_transformer(self):
        return True
    
    @property
    def is_fitted(self) -> bool:
        """Check if transformer is fitted"""
        return self._fitted_state is not None
    
    @property
    def fitted_params(self) -> Dict[str, Any]:
        """Get fitted parameters"""
        if not self.is_fitted:
            raise ValueError(f"Transformer {self.__class__.__name__} is not fitted yet")
        return self._fitted_state.fitted_params
    
    # === Methods to override ===
    
    @staticmethod
    @abstractmethod
    def _inner_function(data: Dataset, **kwargs) -> Dataset:
        """
        Old method for backwards compatibility.
        Used in fit_transform mode.
        """
        raise AbstractMethodError
    
    @staticmethod
    def _fit(data: Dataset, **kwargs) -> Dict[str, Any]:
        """
        Compute transformation parameters from data.
        
        Args:
            data: Input data
            **kwargs: Transformer parameters
        
        Returns:
            Dict with fitted parameters
        
        Override for stateful transformers.
        Default: returns empty dict (stateless).
        """
        return {}
    
    @staticmethod
    def _transform(data: Dataset, fitted_params: Dict[str, Any], **kwargs) -> Dataset:
        """
        Apply transformation using fitted parameters.
        
        Args:
            data: Input data
            fitted_params: Parameters from _fit
            **kwargs: Additional parameters
        
        Returns:
            Transformed Dataset
        
        Override for stateful transformers.
        Default: calls _inner_function (stateless).
        """
        return Transformer._inner_function(data, **kwargs)
    
    # === Public methods ===
    
    @classmethod
    def fit(cls, data: Dataset, **kwargs) -> TransformerState:
        """
        Fit transformer on data.
        
        Returns:
            TransformerState with fitted parameters
        """
        fitted_params = cls._fit(data, **kwargs)
        
        return TransformerState(
            transformer_id="temp",  # will be updated in execute
            transformer_class=cls.__name__,
            fitted_params=fitted_params,
            metadata={"init_kwargs": kwargs}
        )
    
    @classmethod
    def transform(cls, data: Dataset, fitted_params: Dict[str, Any], **kwargs) -> Dataset:
        """
        Apply transformation with fitted parameters.
        """
        return cls._transform(data, fitted_params, **kwargs)
    
    @classmethod
    def fit_transform(cls, data: Dataset, **kwargs) -> tuple[Dataset, TransformerState]:
        """
        Fit and apply transformation.
        
        Returns:
            (transformed_data, state)
        """
        state = cls.fit(data, **kwargs)
        transformed = cls.transform(data, state.fitted_params, **kwargs)
        return transformed, state
    
    @classmethod
    def calc(cls, data: Dataset, **kwargs):
        """Backwards compatibility"""
        return cls._inner_function(data, **kwargs)
    
    def execute(self, data: ExperimentData) -> ExperimentData:
        """
        Execute transformation in selected mode.
        
        Saves/loads state to/from MLExperimentData.transformer_states[self.id]
        """
        # For non-ML Experiments - old behavior
        from ..dataset.ml_data import MLExperimentData
        if not isinstance(data, MLExperimentData):
            return data.copy(data=self.calc(data=data.ds))
        
        # For MLExperiment - delegate to ML execution
        return self._execute_ml(data)
    
    def _execute_ml(self, data: "MLExperimentData") -> "MLExperimentData":
        """Execute in MLExperiment context"""
        
        if self.mode == TransformerMode.FIT_TRANSFORM:
            # Mode 1: Fit + Transform
            transformed_ds, state = self.fit_transform(data.ds, **self.calc_kwargs)
            state.transformer_id = self.id
            self._fitted_state = state
            data.add_fitted_transformer(self.id, state)
            return data.copy(data=transformed_ds)
        
        elif self.mode == TransformerMode.FIT:
            # Mode 2: Only Fit
            state = self.fit(data.ds, **self.calc_kwargs)
            state.transformer_id = self.id
            self._fitted_state = state
            data.add_fitted_transformer(self.id, state)
            return data  # no transformation
        
        elif self.mode == TransformerMode.TRANSFORM:
            # Mode 3: Only Transform
            state = data.get_fitted_transformer(self.id)
            if state is None:
                raise ValueError(
                    f"Transformer {self.__class__.__name__} (id={self.id}) "
                    f"is in 'transform' mode but no fitted state found. "
                    f"Available states: {list(data.ml.get('fitted_transformers', {}).keys())}"
                )
            
            self._fitted_state = state
            transformed_ds = self.transform(data.ds, state.fitted_params, **self.calc_kwargs)
            return data.copy(data=transformed_ds)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
