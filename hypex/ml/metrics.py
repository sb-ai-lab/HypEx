from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..dataset import Dataset, ExperimentData, PredictionRole, TargetRole
from ..executor import Executor
from ..extensions.ml_metrics import MetricExtension
from ..utils import ExperimentDataEnum


class MLMetric(Executor, ABC):
    """Base class for ML metric executors.

    Reads predictions from ``additional_fields`` and target from ``ds``,
    computes a metric via its extension, stores result in ``variables``.

    Each subclass implements :meth:`_inner_function` that instantiates
    the appropriate Extension and calls ``.calc()`` — the same pattern
    used by :class:`~hypex.comparators.abstract.StatHypothesisTesting`.

    Args:
        prediction_column: Explicit column name for predictions in
            ``additional_fields``.  When *None* the last column with
            :class:`PredictionRole` is used automatically.
        key: Optional executor key.
    """

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Short metric identifier used as a dict key (e.g. ``'r2'``)."""

    @property
    def higher_is_better(self) -> bool:
        """Whether higher metric values indicate a better model."""
        return True

    def __init__(self, prediction_column: str | None = None, key: Any = ""):
        self.prediction_column = prediction_column
        super().__init__(key=key)

    def execute(self, data: ExperimentData) -> ExperimentData:
        """Compute the metric and store result in ``variables``.

        Reads predictions and target from *data*, delegates
        computation to :meth:`_inner_function`, and writes the
        scalar value into ``data.variables[self.id][metric_name]``.

        Args:
            data: Experiment data with predictions and target.

        Returns:
            The same *data* with metric value stored in ``variables``.
        """
        # Find predictions
        if self.prediction_column:
            predictions = data.additional_fields[self.prediction_column]
        else:
            stat_cols = data.additional_fields.search_columns(PredictionRole())
            if not stat_cols:
                raise ValueError("No prediction columns found in additional_fields")
            predictions = data.additional_fields[stat_cols[-1]]

        # Find target
        target_cols = data.ds.search_columns(TargetRole())
        target = data.ds[target_cols]

        # Compute via extension
        value = self._inner_function(predictions, target)

        # Store in variables
        data.set_value(
            ExperimentDataEnum.variables, self.id, value, key=self.metric_name
        )
        return data

    @classmethod
    @abstractmethod
    def _inner_function(
        cls, data: Dataset, target: Dataset | None = None, **kwargs: Any
    ) -> float:
        """Compute the metric value via the corresponding Extension.

        Args:
            data: Dataset with predicted values.
            target: Dataset with ground-truth values.
            **kwargs: Additional keyword arguments forwarded to the extension.

        Returns:
            Scalar metric value.
        """
        ...


class R2Score(MLMetric):
    """R-squared (coefficient of determination).

    Measures the proportion of variance in the target explained by
    the model.  Higher is better; maximum value is 1.0.
    """

    @property
    def metric_name(self) -> str:
        """Return ``'r2'``."""
        return "r2"

    @classmethod
    def _inner_function(cls, data: Dataset, target: Dataset | None = None, **kwargs: Any) -> float:
        """Compute R² via :class:`~hypex.extensions.ml_metrics.R2Extension`."""
        from ..extensions.ml_metrics import R2Extension

        return R2Extension().calc(data, target=target, **kwargs)


class MSE(MLMetric):
    """Mean Squared Error.

    Average of squared differences between predictions and target.
    Lower is better.
    """

    @property
    def metric_name(self) -> str:
        """Return ``'mse'``."""
        return "mse"

    @property
    def higher_is_better(self) -> bool:
        """Return *False* — lower MSE is better."""
        return False

    @classmethod
    def _inner_function(cls, data: Dataset, target: Dataset | None = None, **kwargs: Any) -> float:
        """Compute MSE via :class:`~hypex.extensions.ml_metrics.MSEExtension`."""
        from ..extensions.ml_metrics import MSEExtension

        return MSEExtension().calc(data, target=target, **kwargs)


class MAE(MLMetric):
    """Mean Absolute Error.

    Average of absolute differences between predictions and target.
    Lower is better.
    """

    @property
    def metric_name(self) -> str:
        """Return ``'mae'``."""
        return "mae"

    @property
    def higher_is_better(self) -> bool:
        """Return *False* — lower MAE is better."""
        return False

    @classmethod
    def _inner_function(cls, data: Dataset, target: Dataset | None = None, **kwargs: Any) -> float:
        """Compute MAE via :class:`~hypex.extensions.ml_metrics.MAEExtension`."""
        from ..extensions.ml_metrics import MAEExtension

        return MAEExtension().calc(data, target=target, **kwargs)


class RMSE(MLMetric):
    """Root Mean Squared Error.

    Square root of the average squared difference between predictions
    and target.  Lower is better.
    """

    @property
    def metric_name(self) -> str:
        """Return ``'rmse'``."""
        return "rmse"

    @property
    def higher_is_better(self) -> bool:
        """Return *False* — lower RMSE is better."""
        return False

    @classmethod
    def _inner_function(cls, data: Dataset, target: Dataset | None = None, **kwargs: Any) -> float:
        """Compute RMSE via :class:`~hypex.extensions.ml_metrics.RMSEExtension`."""
        from ..extensions.ml_metrics import RMSEExtension

        return RMSEExtension().calc(data, target=target, **kwargs)
