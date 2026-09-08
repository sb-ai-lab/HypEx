from __future__ import annotations

from typing import Any

import pandas as pd

from ..dataset import Dataset, PredictionRole
from .abstract import MLExtension


class SklearnModelExtension(MLExtension):
    """Extension for sklearn-compatible models.

    Encapsulates sklearn dependencies (``clone``, ``fit``, ``predict``)
    and numpy array conversions.  The model prototype is stored
    and cloned for every :meth:`fit` call.

    Args:
        model: An unfitted sklearn-compatible estimator instance.
    """

    def __init__(self, model: Any):
        super().__init__()
        self._model_proto = model

    def fit(self, features: Dataset, target: Dataset | None = None, **kwargs: Any) -> Any:
        """Clone the prototype, fit on *features* / *target*, return the fitted model.

        Args:
            features: Dataset with predictor columns.
            target: Dataset with the target column.
            **kwargs: Ignored.

        Returns:
            Fitted sklearn estimator.
        """
        from sklearn.base import clone

        model = clone(self._model_proto)
        model.fit(features.data.values, target.data.values.ravel())
        return model

    def predict(self, features: Dataset, model: Any = None, **kwargs: Any) -> Dataset:
        """Generate predictions using a previously fitted model.

        Args:
            features: Dataset with predictor columns.
            model: Fitted sklearn estimator (artifact from :meth:`fit`).
            **kwargs: Ignored.

        Returns:
            Dataset with a single ``prediction`` column
            (:class:`PredictionRole`).

        Raises:
            ValueError: If *model* is not provided.
        """
        if model is None:
            raise ValueError("Fitted model is required for prediction")
        preds = model.predict(features.data.values)
        return Dataset(
            roles={"prediction": PredictionRole()},
            data=pd.DataFrame(
                {"prediction": preds},
                index=features.data.index,
            ),
        )

    def _calc_pandas(
        self,
        data: Dataset,
        mode: str | None = None,
        model: Any = None,
        target: Dataset | None = None,
        **kwargs: Any,
    ) -> Any:
        """Dispatch to :meth:`fit` or :meth:`predict` based on *mode*.

        Args:
            data: Dataset with feature columns.
            mode: ``'fit'`` or ``'predict'``.
            model: Fitted estimator (for predict mode).
            target: Target dataset (for fit mode).
            **kwargs: Forwarded to the underlying method.
        """
        if mode == "fit":
            return self.fit(data, target=target, **kwargs)
        return self.predict(data, model=model, **kwargs)


class StandardScalerExtension(MLExtension):
    """Extension for feature standardization (z-score).

    Isolates numpy operations for mean/std computation and scaling.
    """

    def fit(self, features: Dataset, **kwargs: Any) -> dict[str, dict[str, float]]:
        """Compute per-column mean and standard deviation.

        Args:
            features: Dataset with numeric feature columns.
            **kwargs: Ignored.

        Returns:
            Dict ``{'mean': {col: val}, 'std': {col: val}}``.
        """
        import numpy as np

        means = {
            col: float(np.mean(features.data[col].values))
            for col in features.columns
        }
        stds = {
            col: float(np.std(features.data[col].values, ddof=0))
            for col in features.columns
        }
        return {"mean": means, "std": stds}

    def predict(self, features: Dataset, artifact: dict | None = None, **kwargs: Any) -> Dataset:
        """Apply z-score scaling using a previously fitted artifact.

        Columns with zero standard deviation are left unchanged.

        Args:
            features: Dataset to scale.
            artifact: Dict returned by :meth:`fit` with ``'mean'``
                and ``'std'`` per column.
            **kwargs: Ignored.

        Returns:
            Scaled Dataset preserving original roles and index.

        Raises:
            ValueError: If *artifact* is not provided.
        """
        if artifact is None:
            raise ValueError("Fitted artifact is required for transform")
        result_data = features.data.copy()
        for col in features.columns:
            std = artifact["std"][col]
            if std > 0:
                result_data[col] = (
                    result_data[col] - artifact["mean"][col]
                ) / std
        return Dataset(roles=features.roles, data=result_data)

    def _calc_pandas(
        self,
        data: Dataset,
        mode: str | None = None,
        artifact: dict | None = None,
        **kwargs: Any,
    ) -> Any:
        """Dispatch to :meth:`fit` or :meth:`predict` based on *mode*.

        Args:
            data: Dataset with feature columns.
            mode: ``'fit'`` or ``'transform'``.
            artifact: Fitted statistics (for transform mode).
            **kwargs: Forwarded to the underlying method.
        """
        if mode == "fit":
            return self.fit(data, **kwargs)
        return self.predict(data, artifact=artifact, **kwargs)
