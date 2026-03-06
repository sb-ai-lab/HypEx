from __future__ import annotations

from typing import Any, Callable

from ..dataset import Dataset
from .abstract import Extension


class MetricExtension(Extension):
    """Base extension for ML metrics.

    Follows the same pattern as :class:`StatTest`: the constructor
    receives a ``metric_function`` callable (from ``sklearn.metrics``)
    and :meth:`_calc_pandas` simply invokes it.

    Subclasses only need to pass the right function in ``__init__``.

    Args:
        metric_function: Callable ``(y_true, y_pred) -> float``.
    """

    def __init__(self, metric_function: Callable | None = None):
        super().__init__()
        self.metric_function = metric_function

    def _calc_pandas(
        self,
        data: Dataset,
        target: Dataset | None = None,
        **kwargs: Any,
    ) -> float:
        """Compute the metric on pandas-backed Datasets.

        Args:
            data: Dataset with predicted values.
            target: Dataset with ground-truth values.
            **kwargs: Ignored (reserved for backend dispatch).

        Returns:
            Scalar metric value.

        Raises:
            ValueError: If *target* or *metric_function* is missing.
        """
        if target is None:
            raise ValueError("target is required for metric computation")
        if self.metric_function is None:
            raise ValueError("metric_function is needed for evaluation")

        y_true = target.data.values.ravel()
        y_pred = data.data.values.ravel()
        return float(self.metric_function(y_true, y_pred))


class R2Extension(MetricExtension):
    """R-squared (coefficient of determination).

    Wraps ``sklearn.metrics.r2_score``.
    """

    def __init__(self):
        from sklearn.metrics import r2_score

        super().__init__(r2_score)


class MSEExtension(MetricExtension):
    """Mean Squared Error.

    Wraps ``sklearn.metrics.mean_squared_error``.
    """

    def __init__(self):
        from sklearn.metrics import mean_squared_error

        super().__init__(mean_squared_error)


class MAEExtension(MetricExtension):
    """Mean Absolute Error.

    Wraps ``sklearn.metrics.mean_absolute_error``.
    """

    def __init__(self):
        from sklearn.metrics import mean_absolute_error

        super().__init__(mean_absolute_error)


class RMSEExtension(MetricExtension):
    """Root Mean Squared Error.

    Computes ``sqrt(mean_squared_error(y_true, y_pred))``.
    """

    def __init__(self):
        from sklearn.metrics import mean_squared_error

        super().__init__(mean_squared_error)

    def _calc_pandas(
        self,
        data: Dataset,
        target: Dataset | None = None,
        **kwargs: Any,
    ) -> float:
        """Compute RMSE as square root of MSE."""
        return super()._calc_pandas(data, target, **kwargs) ** 0.5
