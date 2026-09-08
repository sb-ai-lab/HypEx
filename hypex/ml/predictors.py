from __future__ import annotations

from typing import Any

from ..dataset import Dataset
from ..extensions.ml_models import SklearnModelExtension
from ..utils import PredictorModeEnum
from .base import MLPredictor


class SklearnPredictor(MLPredictor):
    """Predictor wrapping sklearn-compatible models.

    Delegates fit/predict to :class:`SklearnModelExtension`,
    keeping the executor backend-agnostic.

    Args:
        extension: Extension wrapping the sklearn model.
        mode: Operating mode (``'fit'``, ``'predict'``,
            or ``'fit_predict'``).
        key: Optional executor key.
    """

    def __init__(
        self,
        extension: SklearnModelExtension,
        mode: PredictorModeEnum = PredictorModeEnum.fit_predict,
        key: Any = "",
    ):
        self.extension = extension
        super().__init__(mode=mode, key=key)

    def _fit(self, features: Dataset, target: Dataset) -> Any:
        """Train the model via extension and return the fitted artifact."""
        return self.extension.calc(data=features, mode="fit", target=target)

    def _predict(self, features: Dataset, artifact: Any) -> Dataset:
        """Generate predictions via extension using a fitted artifact."""
        return self.extension.calc(data=features, mode="predict", model=artifact)


class LinearRegressionPredictor(SklearnPredictor):
    """Ordinary least-squares linear regression predictor.

    Args:
        mode: Operating mode.
        key: Optional executor key.
    """

    def __init__(self, mode: PredictorModeEnum = PredictorModeEnum.fit_predict, key: Any = ""):
        from sklearn.linear_model import LinearRegression

        super().__init__(SklearnModelExtension(LinearRegression()), mode, key)


class RidgePredictor(SklearnPredictor):
    """Ridge (L2-regularised) regression predictor.

    Args:
        alpha: Regularisation strength.
        mode: Operating mode.
        key: Optional executor key.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        mode: PredictorModeEnum = PredictorModeEnum.fit_predict,
        key: Any = "",
    ):
        from sklearn.linear_model import Ridge

        super().__init__(SklearnModelExtension(Ridge(alpha=alpha)), mode, key)


class RandomForestPredictor(SklearnPredictor):
    """Random forest regression predictor.

    Args:
        n_estimators: Number of trees in the forest.
        mode: Operating mode.
        key: Optional executor key.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        mode: PredictorModeEnum = PredictorModeEnum.fit_predict,
        key: Any = "",
    ):
        from sklearn.ensemble import RandomForestRegressor

        super().__init__(
            SklearnModelExtension(
                RandomForestRegressor(n_estimators=n_estimators)
            ),
            mode,
            key,
        )


class CatBoostPredictor(SklearnPredictor):
    """CatBoost gradient boosting regression predictor.

    Requires the optional ``catboost`` dependency.

    Args:
        iterations: Maximum number of boosting iterations.
        learning_rate: Step size shrinkage.  *None* uses the
            CatBoost default.
        depth: Maximum tree depth.  *None* uses the CatBoost default.
        verbose: Verbosity level (default ``0`` — silent).
        mode: Operating mode.
        key: Optional executor key.

    Raises:
        ImportError: If *catboost* is not installed.
    """

    def __init__(
        self,
        iterations: int = 100,
        learning_rate: float | None = None,
        depth: int | None = None,
        verbose: int = 0,
        mode: PredictorModeEnum = PredictorModeEnum.fit_predict,
        key: Any = "",
    ):
        try:
            from catboost import CatBoostRegressor
        except ImportError as e:
            raise ImportError(
                "catboost is required for CatBoostPredictor. "
                "Install it with: pip install catboost"
            ) from e

        params: dict[str, Any] = {"iterations": iterations, "verbose": verbose}
        if learning_rate is not None:
            params["learning_rate"] = learning_rate
        if depth is not None:
            params["depth"] = depth

        super().__init__(
            SklearnModelExtension(CatBoostRegressor(**params)),
            mode,
            key,
        )
