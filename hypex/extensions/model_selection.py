from __future__ import annotations

from typing import Any

from ..dataset import Dataset
from .abstract import Extension

# Mapping: our metric name -> sklearn scoring name
_SKLEARN_SCORING: dict[str, str] = {
    "r2": "r2",
    "mse": "neg_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
}

# Metrics where sklearn returns negative values (need to negate for display)
_NEGATE_METRICS: set[str] = {"mse", "mae", "rmse"}


class CrossValidationExtension(Extension):
    """Cross-validates multiple models with multiple metrics.

    Uses sklearn.model_selection.cross_validate under the hood.
    Converts sklearn neg_ scoring back to positive values.
    """

    def _calc_pandas(
        self,
        data: Dataset,
        target: Dataset | None = None,
        models: dict[str, Any] | None = None,
        cv: int = 5,
        metrics: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Run cross-validation for every model × metric combination.

        Args:
            data: Dataset with feature columns.
            target: Dataset with the target column.
            models: ``{name: sklearn_estimator}`` mapping.
            cv: Number of cross-validation folds.
            metrics: List of metric identifiers
                (``'r2'``, ``'mse'``, ``'mae'``, ``'rmse'``).
            **kwargs: Ignored.

        Returns:
            Nested dict ``{model_name: {metric: {'mean': …, 'std': …}}}``.
        """
        from sklearn.base import clone
        from sklearn.model_selection import cross_validate

        features_arr = data.data.values
        target_arr = target.data.values.ravel()
        metrics = metrics or ["r2"]

        scoring = {m: _SKLEARN_SCORING[m] for m in metrics}

        results: dict[str, dict[str, dict[str, float]]] = {}
        for name, model in (models or {}).items():
            cv_res = cross_validate(
                clone(model),
                features_arr,
                target_arr,
                cv=cv,
                scoring=scoring,
            )
            results[name] = {}
            for metric in metrics:
                scores = cv_res[f"test_{metric}"]
                if metric in _NEGATE_METRICS:
                    scores = -scores
                results[name][metric] = {
                    "mean": float(scores.mean()),
                    "std": float(scores.std()),
                }
        return results
