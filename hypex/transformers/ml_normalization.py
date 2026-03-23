from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

from ..dataset import Dataset
from ..dataset.ml_data import MLExperimentData
from ..dataset.roles import FeatureRole
from ..utils import NUMBER_TYPES_LIST
from .ml_transformer import MLTransformer


class MLNormalizationTransformer(MLTransformer):
    """Normalize numeric features using fit/predict lifecycle.

    Supported strategies:
    - ``minmax``: ``(x - min) / (max - min)``
    - ``standard``: ``(x - mean) / std``
    """

    _SUPPORTED_STRATEGIES = {"minmax", "standard"}

    def __init__(
        self,
        target_roles: str | Sequence[str] | None = None,
        strategy: str = "standard",
        epsilon: float = 1e-8,
        key: Any = "",
    ):
        if strategy not in self._SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Supported: "
                f"{sorted(self._SUPPORTED_STRATEGIES)}"
            )
        if epsilon <= 0:
            raise ValueError("epsilon must be greater than 0")

        super().__init__(key=key, strategy=strategy, epsilon=epsilon)
        self.target_roles = target_roles or FeatureRole()

    @property
    def search_types(self):
        return NUMBER_TYPES_LIST

    @staticmethod
    def _extract_scalar(value: Dataset | int | float | None) -> float | None:
        if isinstance(value, Dataset):
            if value.data.empty:
                return None
            raw = value.data.iloc[0, 0]
            if raw is None:
                return None
            return float(raw)
        if value is None:
            return None
        return float(value)

    @classmethod
    def _fit(
        cls,
        data: Dataset,
        target_cols: Sequence[str] | None = None,
        strategy: str = "standard",
        **kwargs,
    ) -> dict[str, Any]:
        target_cols = list(target_cols or [])
        params_by_col: dict[str, dict[str, float | None]] = {}

        for column in target_cols:
            if column not in data.columns:
                continue

            col_data = data[[column]]
            if strategy == "minmax":
                min_val = cls._extract_scalar(col_data.min())
                max_val = cls._extract_scalar(col_data.max())
                params_by_col[column] = {"min": min_val, "max": max_val}
            else:
                mean_val = cls._extract_scalar(col_data.mean())
                std_val = cls._extract_scalar(col_data.std())
                params_by_col[column] = {"mean": mean_val, "std": std_val}

        return {
            "target_cols": [c for c in target_cols if c in params_by_col],
            "strategy": strategy,
            "params": params_by_col,
        }

    @staticmethod
    def _transform(
        data: Dataset,
        fitted_params: dict[str, Any],
        epsilon: float = 1e-8,
        **kwargs,
    ) -> Dataset:
        target_cols = fitted_params.get("target_cols", [])
        strategy = fitted_params.get("strategy", "standard")
        params = fitted_params.get("params", {})

        transformed_df = data.data.copy(deep=True)
        transformed_roles = deepcopy(data.roles)

        for column in target_cols:
            if column not in transformed_df.columns or column not in params:
                continue

            column_params = params[column]
            col_series = transformed_df[column]

            if strategy == "minmax":
                min_val = column_params.get("min")
                max_val = column_params.get("max")
                if min_val is None or max_val is None:
                    continue
                denominator = max_val - min_val
                if abs(denominator) <= epsilon:
                    transformed_df[column] = col_series * 0
                else:
                    transformed_df[column] = (col_series - min_val) / denominator
            else:
                mean_val = column_params.get("mean")
                std_val = column_params.get("std")
                if mean_val is None or std_val is None:
                    continue
                if abs(std_val) <= epsilon:
                    transformed_df[column] = col_series * 0
                else:
                    transformed_df[column] = (col_series - mean_val) / std_val

            transformed_roles[column] = transformed_roles[column].astype(float)

        return Dataset(data=transformed_df, roles=transformed_roles)

    def execute_fit(self, data: MLExperimentData) -> MLExperimentData:
        target_cols = data.ds.search_columns(
            roles=self.target_roles,
            search_types=self.search_types,
        )
        self.calc_kwargs["target_cols"] = target_cols
        return super().execute_fit(data)

    def execute_fit_predict(self, data: MLExperimentData) -> MLExperimentData:
        target_cols = data.ds.search_columns(
            roles=self.target_roles,
            search_types=self.search_types,
        )
        self.calc_kwargs["target_cols"] = target_cols
        return super().execute_fit_predict(data)
