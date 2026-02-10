"""Transformer state management for MLExperiment."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


class TransformerState:
    """
    Stores fitted parameters of a transformer.
    
    Similar to ModelStats for ML models.
    Transformers fit on training data and save state here,
    then use this state to transform test data.
    
    Args:
        transformer_id: Unique identifier (from executor.id)
        transformer_class: Class name of transformer
        fitted_params: Parameters computed during fit (means, stds, thresholds, etc.)
        metadata: Additional metadata
    
    Examples:
        >>> # StandardScaler saves means and stds
        >>> state = TransformerState(
        ...     transformer_id="StandardScaler_abc123",
        ...     transformer_class="StandardScaler",
        ...     fitted_params={
        ...         "means": {"col1": 0.5, "col2": 1.2},
        ...         "stds": {"col1": 0.2, "col2": 0.8}
        ...     }
        ... )
    """
    
    def __init__(
        self,
        transformer_id: str,
        transformer_class: str,
        fitted_params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.transformer_id = transformer_id
        self.transformer_class = transformer_class
        self.fitted_params = fitted_params
        self.metadata = metadata or {}
        self.fitted_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON storage"""
        return {
            "transformer_id": self.transformer_id,
            "transformer_class": self.transformer_class,
            "fitted_params": self.fitted_params,
            "metadata": self.metadata,
            "fitted_at": self.fitted_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TransformerState:
        """Deserialize from dict"""
        state = cls(
            transformer_id=data["transformer_id"],
            transformer_class=data["transformer_class"],
            fitted_params=data["fitted_params"],
            metadata=data.get("metadata"),
        )
        state.fitted_at = data.get("fitted_at")
        return state
    
    def __repr__(self) -> str:
        return (
            f"TransformerState(id={self.transformer_id}, "
            f"class={self.transformer_class}, "
            f"params={list(self.fitted_params.keys())})"
        )
