from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class CVStats:
    """
    Statistics from cross-validation model training.
    
    Tracks performance metrics across folds for model selection and evaluation.
    
    Attributes:
        variance_reduction: Mean variance reduction across folds (%).
        feature_importances: Mean feature importance values across folds.
        n_folds: Number of cross-validation folds used.
        fold_scores: Individual fold scores (optional).
    """
    variance_reduction: float
    feature_importances: dict[str, float]
    n_folds: int
    fold_scores: Optional[list[float]] = None
    
    def __post_init__(self):
        """Validate statistics values."""
        if self.variance_reduction < 0:
            raise ValueError("Variance reduction cannot be negative")
        if self.n_folds < 2:
            raise ValueError("Number of folds must be at least 2")


@dataclass
class ModelStats:
    """
    Comprehensive statistics for a trained model.
    
    Stores both cross-validation metrics and real-world performance metrics
    for a model applied to actual data.
    
    Attributes:
        model_name: Name/identifier of the model.
        cv_stats: Cross-validation statistics.
        variance_reduction_real: Variance reduction on actual data (%).
        additional_metrics: Dictionary for any additional model metrics.
    """
    model_name: str
    cv_stats: CVStats
    variance_reduction_real: Optional[float] = None
    additional_metrics: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert model statistics to a dictionary format.
        
        Returns:
            Dictionary with all statistics in a flat structure for reporting.
        """
        result = {
            'model_name': self.model_name,
            'variance_reduction_cv': self.cv_stats.variance_reduction,
            'variance_reduction_real': self.variance_reduction_real,
            'feature_importances': self.cv_stats.feature_importances,
            'n_folds': self.cv_stats.n_folds,
        }
        
        if self.cv_stats.fold_scores is not None:
            result['fold_scores'] = self.cv_stats.fold_scores
        
        result.update(self.additional_metrics)
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelStats:
        """
        Create ModelStats instance from a dictionary.
        
        Args:
            data: Dictionary containing model statistics.
            
        Returns:
            ModelStats instance.
        """
        cv_stats = CVStats(
            variance_reduction=data['variance_reduction_cv'],
            feature_importances=data['feature_importances'],
            n_folds=data.get('n_folds', 5),
            fold_scores=data.get('fold_scores')
        )
        
        additional_metrics = {
            k: v for k, v in data.items()
            if k not in ['model_name', 'variance_reduction_cv', 'variance_reduction_real',
                        'feature_importances', 'n_folds', 'fold_scores']
        }
        
        return cls(
            model_name=data['model_name'],
            cv_stats=cv_stats,
            variance_reduction_real=data.get('variance_reduction_real'),
            additional_metrics=additional_metrics
        )


@dataclass
class MLExecutionStats:
    """
    Statistics for an entire ML execution workflow.
    
    Tracks statistics for multiple targets and models, providing
    a comprehensive view of the ML pipeline execution.
    
    Attributes:
        target_stats: Dictionary mapping target names to their best model statistics.
        all_model_attempts: Optional dictionary tracking all model attempts per target.
    """
    target_stats: dict[str, ModelStats]
    all_model_attempts: Optional[dict[str, list[ModelStats]]] = None
    
    def get_best_models(self) -> dict[str, str]:
        """
        Get the best model name for each target.
        
        Returns:
            Dictionary mapping target names to their best model names.
        """
        return {
            target: stats.model_name
            for target, stats in self.target_stats.items()
        }
    
    def get_variance_reductions(self, metric: str = 'cv') -> dict[str, float]:
        """
        Get variance reduction values for all targets.
        
        Args:
            metric: Which variance reduction to return ('cv' or 'real').
            
        Returns:
            Dictionary mapping target names to variance reduction values.
        """
        if metric == 'cv':
            return {
                target: stats.cv_stats.variance_reduction
                for target, stats in self.target_stats.items()
            }
        elif metric == 'real':
            return {
                target: stats.variance_reduction_real
                for target, stats in self.target_stats.items()
                if stats.variance_reduction_real is not None
            }
        else:
            raise ValueError(f"Invalid metric: {metric}. Must be 'cv' or 'real'")
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert execution statistics to dictionary format.
        
        Returns:
            Nested dictionary with all execution statistics.
        """
        result = {
            'target_stats': {
                target: stats.to_dict()
                for target, stats in self.target_stats.items()
            }
        }
        
        if self.all_model_attempts is not None:
            result['all_model_attempts'] = {
                target: [stats.to_dict() for stats in model_stats]
                for target, model_stats in self.all_model_attempts.items()
            }
        
        return result
