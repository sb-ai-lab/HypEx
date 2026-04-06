from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

from ..dataset import Dataset
from ..dataset.ml_data import MLExperimentData
from ..dataset.roles import FeatureRole
from ..utils import CategoricalTypes, NUMBER_TYPES_LIST
from .ml_transformer import MLTransformer


class MLFillnaTransformer(MLTransformer):
    """Fill missing values using fit/predict lifecycle.

    Supported strategies:
    - Categorical: ``mode``, ``const``
    - Numeric: ``mean``, ``median``, ``zero``
    """

    _SUPPORTED_STRATEGIES = {"mode", "const", "mean", "median", "zero"}

    def __init__(
        self,
        target_roles: str | Sequence[str] | None = None,
        strategy: str = "mean",
        fill_value: Any | None = None,
        key: Any = "",
    ):
        if strategy not in self._SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Supported: "
                f"{sorted(self._SUPPORTED_STRATEGIES)}"
            )

        if strategy == "const" and fill_value is None:
            raise ValueError("fill_value is required for strategy='const'")

        super().__init__(key=key, strategy=strategy, fill_value=fill_value)
        self.target_roles = target_roles or FeatureRole()

    @property
    def search_types(self):
        strategy = self.calc_kwargs["strategy"]
        if strategy in {"mode", "const"}:
            return [CategoricalTypes]
        return NUMBER_TYPES_LIST

    @staticmethod
    def _extract_scalar(value: Dataset | int | float | str | bool | None) -> Any:
        if isinstance(value, Dataset):
            if value.data.empty:
                return None
            return value.data.iloc[0, 0]
        return value

    @classmethod
    def _compute_fill_value(
        cls,
        data: Dataset,
        column: str,
        strategy: str,
        fill_value: Any | None,
    ) -> Any | None:
        col_data = data[[column]]

        if strategy == "const":
            return fill_value

        if strategy == "zero":
            return 0

        if strategy == "mean":
            return cls._extract_scalar(col_data.mean())

        if strategy == "median":
            return cls._extract_scalar(col_data.quantile(0.5))

        # strategy == "mode"
        mode_ds = col_data.mode(dropna=True)
        if mode_ds.data.empty:
            return fill_value
        return mode_ds.data.iloc[0, 0]

    @classmethod
    def _fit(
        cls,
        data: Dataset,
        target_cols: Sequence[str] | None = None,
        strategy: str = "mean",
        fill_value: Any | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        target_cols = list(target_cols or [])
        values_by_col: dict[str, Any] = {}

        for column in target_cols:
            if column not in data.columns:
                continue
            value = cls._compute_fill_value(data, column, strategy, fill_value)
            if value is None and strategy != "const":
                continue
            values_by_col[column] = value

        return {
            "target_cols": [c for c in target_cols if c in values_by_col],
            "strategy": strategy,
            "fill_values": values_by_col,
        }

    @staticmethod
    def _transform(
        data: Dataset,
        fitted_params: dict[str, Any],
        **kwargs,
    ) -> Dataset:
        target_cols = fitted_params.get("target_cols", [])
        fill_values = fitted_params.get("fill_values", {})

        transformed_df = data.data.copy(deep=True)
        transformed_roles = deepcopy(data.roles)

        for column in target_cols:
            if column not in transformed_df.columns or column not in fill_values:
                continue

            value = fill_values[column]
            transformed_df[column] = transformed_df[column].fillna(value)

            if isinstance(value, float):
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
