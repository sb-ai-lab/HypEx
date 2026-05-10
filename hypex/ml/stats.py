from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


class ModelStats:
    """Statistics for a trained model with optional CV results"""
    
    def __init__(
        self,
        model_name: str,
        model_type: str,
        feature_importances: Dict[str, float],
        training_time_seconds: float,
        cv_score: Optional[float] = None,
        cv_scores_per_fold: Optional[List[float]] = None,
        n_folds: Optional[int] = None,
        variance_reduction_cv: Optional[float] = None,
        variance_reduction_real: Optional[float] = None,
        aggregation_method: Optional[str] = None,
        **metadata,
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.cv_score = cv_score
        self.cv_scores_per_fold = cv_scores_per_fold
        self.feature_importances = feature_importances
        self.variance_reduction_cv = variance_reduction_cv or cv_score
        self.variance_reduction_real = variance_reduction_real
        self.n_folds = n_folds
        self.training_time_seconds = training_time_seconds
        self.aggregation_method = aggregation_method
        self.timestamp = datetime.now().isoformat()
        self.metadata = metadata
    
    def to_dict(self) -> Dict:
        """Serialize to dict"""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "cv_score": self.cv_score,
            "cv_scores_per_fold": self.cv_scores_per_fold,
            "feature_importances": self.feature_importances,
            "variance_reduction_cv": self.variance_reduction_cv,
            "variance_reduction_real": self.variance_reduction_real,
            "n_folds": self.n_folds,
            "training_time_seconds": self.training_time_seconds,
            "aggregation_method": self.aggregation_method,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> ModelStats:
        """Deserialize from dict"""
        return cls(
            model_name=data["model_name"],
            model_type=data["model_type"],
            feature_importances=data["feature_importances"],
            training_time_seconds=data["training_time_seconds"],
            cv_score=data.get("cv_score"),
            cv_scores_per_fold=data.get("cv_scores_per_fold"),
            n_folds=data.get("n_folds"),
            variance_reduction_cv=data.get("variance_reduction_cv"),
            variance_reduction_real=data.get("variance_reduction_real"),
            aggregation_method=data.get("aggregation_method"),
            **data.get("metadata", {}),
        )
    
    def __repr__(self) -> str:
        cv_str = f"{self.cv_score:.2f}" if self.cv_score is not None else "N/A"
        real_str = (
            f"{self.variance_reduction_real:.2f}"
            if self.variance_reduction_real is not None
            else "N/A"
        )
        return (
            f"ModelStats(model={self.model_name}, "
            f"cv_score={cv_str}, "
            f"var_red_real={real_str})"
        )
