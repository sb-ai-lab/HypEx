from __future__ import annotations

from typing import Any

from ..dataset import Dataset, FeatureRole
from ..extensions.ml_models import StandardScalerExtension
from ..utils import TransformerModeEnum
from .base import MLTransformer


class StandardScaler(MLTransformer):
    """Standardize features by removing the mean and scaling to unit variance.

    Delegates computation to
    :class:`~hypex.extensions.ml_models.StandardScalerExtension`.

    Only numeric columns with :class:`FeatureRole` are scaled.

    Args:
        mode: Operating mode (``'fit'``, ``'transform'``,
            or ``'fit_transform'``).
        key: Optional executor key.
    """

    def __init__(self, mode: TransformerModeEnum = TransformerModeEnum.fit_transform, key: Any = ""):
        self.extension = StandardScalerExtension()
        super().__init__(mode=mode, key=key)

    def _fit(self, data: Dataset) -> dict:
        """Compute per-column mean and std for numeric feature columns.

        Args:
            data: Input dataset.

        Returns:
            Dict with ``'mean'`` and ``'std'`` mappings per column.
        """
        feature_cols = data.search_columns(FeatureRole(), search_types=[int, float])
        return self.extension.calc(data=data[feature_cols], mode="fit")

    def _transform(self, data: Dataset, artifact: dict) -> Dataset:
        """Apply z-score scaling using previously fitted statistics.

        Args:
            data: Input dataset.
            artifact: Dict returned by :meth:`_fit`.

        Returns:
            Dataset with scaled feature columns.
        """
        feature_cols = list(artifact["mean"].keys())
        scaled = self.extension.calc(
            data=data[feature_cols], mode="transform", artifact=artifact
        )
        result = data.drop(columns=feature_cols)
        return result.append(scaled, axis=1)
