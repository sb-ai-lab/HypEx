from __future__ import annotations

from sklearn.feature_selection import mutual_info_regression

from ..dataset import Dataset, DatasetAdapter, InfoRole
from .abstract import Extension


class MutualInfoExtension(Extension):
    """Calculate mutual information between features and target."""

    def __init__(self, random_state: int = 42):
        super().__init__()
        self.random_state = random_state

    def _calc_pandas(
        self, data: Dataset, target: Dataset | None = None, **kwargs
    ) -> Dataset:
        """
        Calculate mutual information for each feature.

        Args:
            data: Features Dataset
            target: Target Dataset (single column)

        Returns:
            Dataset with feature names and their MI scores
        """
        if target is None:
            raise ValueError("target Dataset is required")

        feature_names = list(data.columns)
        X = data.backend.data.values
        y = target.backend.data.values.ravel()

        # Handle NaN values
        import numpy as np

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]

        mi_scores = mutual_info_regression(
            X_clean, y_clean, random_state=self.random_state
        )

        result = {
            "feature": feature_names,
            "score": mi_scores.tolist(),
        }
        return DatasetAdapter.to_dataset(result, roles=InfoRole())
