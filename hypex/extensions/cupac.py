from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

from ..dataset import AdditionalTargetRole, Dataset
from ..utils.cuped_theta import cuped_theta
from ..utils.models import CUPAC_MODELS
from .abstract import MLExtension


class CupacExtension(MLExtension):
    def __init__(
        self,
        n_folds: int = 5,
        random_state: int | None = None,
    ):
        super().__init__()
        self.n_folds = n_folds
        self.random_state = random_state

    def calc(
        self,
        data: Dataset,
        mode: Literal["auto", "kfold_fit", "fit", "predict"] | None = None,
        model: str | Any = None,
        Y: Dataset | None = None,
        **kwargs,
    ) -> Any:
        """Route to the appropriate internal method based on mode.

        Args:
            data: Feature dataset (X).
            mode: Operation mode. One of "kfold_fit", "fit", "predict".
            model: Model name (str) for fit modes, or fitted model
                object for predict.
            Y: Target dataset. Required for "kfold_fit" and "fit".
            **kwargs: Additional arguments (unused).

        Returns:
            - "kfold_fit": (variance_reduction, feature_importances)
            - "fit": fitted model object
            - "predict": Dataset with predictions
        """
        if mode == "kfold_fit":
            return self._kfold_fit_pandas(model, data, Y)
        if mode == "fit":
            return self._fit_pandas(model, data, Y)
        if mode == "predict":
            return self._predict_pandas(model, data)
        raise ValueError(f"Unsupported mode: {mode!r}")

    def fit(self, X: Dataset, Y: Dataset | None = None, **kwargs) -> Any:
        """Train model on full data. Delegates to _fit_pandas.

        Args:
            X: Feature dataset.
            Y: Target dataset.
            **kwargs: Must contain 'model' key.

        Returns:
            Fitted model object.
        """
        model = kwargs.get("model")
        return self._fit_pandas(model, X, Y)

    def predict(self, X: Dataset, **kwargs) -> Dataset:
        """Generate predictions using a fitted model.

        Args:
            X: Feature dataset.
            **kwargs: Must contain 'model' key (fitted model object).

        Returns:
            Dataset with a 'predict' column.
        """
        model = kwargs.get("model")
        return self._predict_pandas(model, X)

    def _calc_pandas(
        self,
        data: Dataset,
        mode: Literal["kfold_fit", "fit", "predict"],
        model: str | Any,
        Y: Dataset | None = None,
        **kwargs,
    ) -> Any:
        if mode == "kfold_fit":
            return self._kfold_fit_pandas(model, data, Y)
        if mode == "fit":
            return self._fit_pandas(model, data, Y)
        elif mode == "predict":
            return self._predict_pandas(model, data)
        return None

    def _kfold_fit_pandas(
        self, model: str, X: Dataset, Y: Dataset
    ) -> tuple[float, dict[str, float]]:
        model_proto = CUPAC_MODELS[model]["pandasdataset"]
        X_df = X.raw_data
        Y_df = Y.raw_data
        y_values = Y_df.iloc[:, 0] if len(Y_df.columns) > 0 else Y_df
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        fold_feature_importances = []
        feature_names = X_df.columns.tolist()
        # OOF cross-fitting: theta estimated on out-of-fold predictions
        y_original = y_values.to_numpy()
        oof_pred = np.full(len(y_original), np.nan)
        for train_idx, val_idx in kf.split(X_df):
            X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
            y_train = y_values.iloc[train_idx]
            m = clone(model_proto)
            m.fit(X_train, y_train)
            oof_pred[val_idx] = m.predict(X_val)
            fold_importances = self._extract_fold_importances(m, model, feature_names)
            fold_feature_importances.append(fold_importances)

        theta = cuped_theta(y_original, oof_pred)
        y_adjusted = y_original - theta * (oof_pred - oof_pred.mean())
        mean_var_reduction = self._calculate_variance_reduction(y_original, y_adjusted)

        mean_importances = {
            feature: float(
                np.mean([fold_imp[feature] for fold_imp in fold_feature_importances])
            )
            for feature in feature_names
        }
        return mean_var_reduction, mean_importances

    def _fit_pandas(self, model: str, X: Dataset, Y: Dataset) -> Any:
        model_proto = CUPAC_MODELS[model]["pandasdataset"]
        final_model = clone(model_proto)
        X_df = X.raw_data
        Y_df = Y.raw_data
        y_values = Y_df.iloc[:, 0] if len(Y_df.columns) > 0 else Y_df
        final_model.fit(X_df, y_values)
        return final_model

    def _predict_pandas(self, model: Any, X: Dataset) -> Dataset:
        """Make predictions using pandas backend."""
        X_df = X.raw_data
        predictions = pd.DataFrame(model.predict(X_df), columns=["predict"])
        return Dataset(roles={"predict": AdditionalTargetRole()}, data=predictions)

    @staticmethod
    def _extract_fold_importances(
        model: Any, model_name: str, feature_names: list[str]
    ) -> dict[str, float]:
        """
        Extract feature importances from a fitted model for a single fold.

        Args:
            model: Fitted model object.
            model_name: Model type ('linear', 'ridge', 'lasso', 'catboost').
            feature_names: List of feature names.

        Returns:
            dict: Feature name to importance mapping.
        """
        importances = {}

        if model_name in ["linear", "ridge", "lasso"]:
            for i, feature_name in enumerate(feature_names):
                importances[feature_name] = float(model.coef_[i])
        elif model_name == "catboost":
            for i, feature_name in enumerate(feature_names):
                importances[feature_name] = float(model.feature_importances_[i])

        return importances

    @staticmethod
    def _calculate_variance_reduction(y_original, y_adjusted) -> float:
        """Calculate variance reduction between original and adjusted target."""
        var_original = y_original.var()
        var_adjusted = y_adjusted.var()
        if var_original < 1e-10:
            return 0.0
        return float(max(0, (1 - var_adjusted / var_original) * 100))
