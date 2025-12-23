from __future__ import annotations
from typing import Any, Sequence, Optional, Dict, Union, Literal
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from ..dataset import Dataset, TargetRole, AdditionalTargetRole
from ..dataset.backends import PandasDataset
from .abstract import MLExtension
from ..ml.models import MODEL_REGISTRY, MLModel


class CupacExtension(MLExtension):

    def __init__(
        self,
        n_folds: int = 5,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.n_folds = n_folds
        self.random_state = random_state

    def _calc_pandas(
        self,
        data: Dataset,
        mode: Literal["kfold_fit", "fit", "predict"],
        model: str | Any,
        Y: Dataset | None = None,
        **kwargs) -> Any:
        if mode == 'kfold_fit':
            return self._kfold_fit_pandas(model, data, Y)
        if mode == 'fit':
            return self._fit_pandas(model, data, Y)
        elif mode == 'predict':
            return self._predict_pandas(model, data)

    def fit(self, model: str, X: Dataset, Y: Dataset) -> Any:
        pass

    def predict(self, model: Any, X: Dataset) -> Dataset:
        pass

    def _kfold_fit_pandas(self, model: str, X: Dataset, Y: Dataset) -> tuple[float, dict[str, float]]:
        """
        Perform K-fold cross-validation and return variance reduction and feature importances.
        
        Returns:
            tuple: (mean_variance_reduction, mean_feature_importances)
        """
        ml_model = MODEL_REGISTRY.get_model(model, backend='pandasdataset')
        if ml_model is None:
            raise ValueError(f"Model '{model}' not available for pandasdataset backend")
        
        X_df = X.data
        Y_df = Y.data
        
        y_values = Y_df.iloc[:, 0] if len(Y_df.columns) > 0 else Y_df
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        fold_var_reductions = []
        fold_feature_importances = []
        
        feature_names = X_df.columns.tolist()
        
        for train_idx, val_idx in kf.split(X_df):
            X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
            y_train, y_val = y_values.iloc[train_idx], y_values.iloc[val_idx]
            
            # Clone the MLModel instance
            m = ml_model.clone()
            m.fit(X_train, y_train)
            
            pred = m.predict(X_val)
            
            y_original = y_val.to_numpy()
            y_adjusted = y_original - pred + y_train.mean()
            
            var_reduction = self._calculate_variance_reduction(y_original, y_adjusted)
            fold_var_reductions.append(var_reduction)
            
            # Extract feature importances for this fold using MLModel method
            fold_importances = m.get_feature_importances(feature_names)
            fold_feature_importances.append(fold_importances)
        
        mean_var_reduction = float(np.nanmean(fold_var_reductions))
        
        # Average feature importances across folds: convert to dict with mean values
        mean_importances = {
            feature: float(np.mean([fold_imp[feature] for fold_imp in fold_feature_importances]))
            for feature in feature_names
        }
        
        return mean_var_reduction, mean_importances
    
    def _fit_pandas(self, model: str, X: Dataset, Y: Dataset) -> Any:
        ml_model = MODEL_REGISTRY.get_model(model, backend='pandasdataset')
        if ml_model is None:
            raise ValueError(f"Model '{model}' not available for pandasdataset backend")
        
        final_model = ml_model.clone()
        X_df = X.data
        Y_df = Y.data
        y_values = Y_df.iloc[:, 0] if len(Y_df.columns) > 0 else Y_df
        final_model.fit(X_df, y_values)
        return final_model

    def _predict_pandas(self, model: Any, X: Dataset) -> Dataset:
        """Make predictions using pandas backend."""
        X_df = X.data
        predictions = pd.DataFrame(model.predict(X_df), columns=['predict'])
        return Dataset(
            roles={
                'predict': AdditionalTargetRole()
            },
            data=predictions
        )
    
    @staticmethod
    def _calculate_variance_reduction(y_original, y_adjusted) -> float:
        """Calculate variance reduction between original and adjusted target."""
        var_original = y_original.var()
        var_adjusted = y_adjusted.var()
        if var_original < 1e-10:
            return 0.0
        return float(max(0, (1 - var_adjusted / var_original) * 100))
