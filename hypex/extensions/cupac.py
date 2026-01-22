from __future__ import annotations
from typing import Any, Sequence, Optional, Dict
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from ..dataset import Dataset, TargetRole
from ..dataset.backends import PandasDataset
from .abstract import MLExtension
from ..utils.models import CUPAC_MODELS


class CupacExtension(MLExtension):
    """
    Extension for CUPAC variance reduction using backend-specific implementations.
    """

    def __init__(
        self,
        cupac_features: Dict[str, Sequence[str]],
        available_models: Dict[str, Any],
        explicit_models: list[str],
        n_folds: int = 5,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.cupac_features = cupac_features
        self.available_models = available_models
        self.explicit_models = explicit_models
        self.n_folds = n_folds
        self.random_state = random_state
        self.fitted_models = {}
        self.best_model_names = {}
        self.variance_reductions = {}
        self.is_fitted = False

    def _calc_pandas(
        self,
        data: Dataset,
        mode: str = "auto",
        **kwargs,
    ):
        """Pandas-specific implementation of CUPAC."""
        if mode in ["auto", "fit"]:
            return self._fit_pandas(data)
        elif mode == "predict":
            return self._predict_pandas(data)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _fit_pandas(self, data: Dataset):
        """Fit models using pandas backend."""
        df = data.data.copy()

        self.fitted_models = {}
        self.best_model_names = {}

        for target_feature, pre_target_features in self.cupac_features.items():
            X_cov = df[pre_target_features]
            y = df[target_feature]
            kf = KFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.random_state
            )

            if len(self.explicit_models) == 1:
                model_name = self.explicit_models[0]
                model_proto = self.available_models[model_name]
                model = clone(model_proto)
                model.fit(X_cov, y)
                self.fitted_models[target_feature] = model
                self.best_model_names[target_feature] = model_name
                # Calculate variance reduction for the model
                pred = model.predict(X_cov)
                self.variance_reductions[target_feature] = (
                    self._calculate_variance_reduction(y.to_numpy(), pred)
                )
                continue

            best_score = -np.inf
            best_model = None
            best_model_name = None

            for name in self.explicit_models:
                model_proto = self.available_models[name]
                fold_var_reductions = []
                for train_idx, val_idx in kf.split(X_cov):
                    X_train, X_val = X_cov.iloc[train_idx], X_cov.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    m = clone(model_proto)
                    m.fit(X_train, y_train)
                    pred = m.predict(X_val)
                    fold_var_reductions.append(
                        self._calculate_variance_reduction(y_val.to_numpy(), pred)
                    )
                score = float(np.nanmean(fold_var_reductions))
                if score > best_score:
                    best_score = score
                    best_model = clone(model_proto)
                    best_model_name = name

            if best_model is None:
                raise RuntimeError("No model was selected during model search")
            best_model.fit(X_cov, y)
            self.fitted_models[target_feature] = best_model
            self.best_model_names[target_feature] = best_model_name
            # Calculate variance reduction for the best model
            pred = best_model.predict(X_cov)
            self.variance_reductions[target_feature] = (
                self._calculate_variance_reduction(y.to_numpy(), pred)
            )

        self.is_fitted = True
        return self

    def _predict_pandas(self, data: Dataset) -> Dict[str, Any]:
        """Make predictions using pandas backend."""
        df = data.data.copy()
        result = {}
        for target_feature, pre_target_features in self.cupac_features.items():
            model = self.fitted_models.get(target_feature)
            if model is None:
                raise RuntimeError(
                    f"Model for {target_feature} not fitted. Call fit() first."
                )
            X_cov = df[pre_target_features]
            y = df[target_feature]
            pred = model.predict(X_cov)
            y_adj = y - pred + np.mean(y)
            result[f"{target_feature}_cupac"] = y_adj
        return result

    def fit(self, X: Dataset, Y: Dataset = None) -> "CupacExtension":
        return super().calc(X, mode="fit")

    def predict(self, X: Dataset) -> Dict[str, Any]:
        return super().calc(X, mode="predict")

    def calc(self, data: Dataset, **kwargs):
        self.fit(data)
        return self.predict(data)

    def get_variance_reductions(self):
        return {
            f"{target}_cupac_variance_reduction": reduction
            for target, reduction in self.variance_reductions.items()
        }

    @staticmethod
    def _calculate_variance_reduction(y, pred):
        pred_centered = pred - np.mean(pred)
        if np.var(pred_centered) < 1e-10:
            return 0.0
        theta = np.cov(y, pred_centered)[0, 1] / np.var(pred_centered)
        y_adj = y - theta * pred_centered
        return float(max(0, (1 - np.var(y_adj) / np.var(y)) * 100))
