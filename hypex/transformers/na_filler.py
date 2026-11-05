from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any, Literal

from ..dataset.dataset import Dataset
from ..dataset.experiment_data import ExperimentData
from ..dataset.roles import ABCRole, FeatureRole
from ..utils import ScalarType
from ..utils.adapter import Adapter
from .abstract import MLtransformer, Transformer


class NaFiller(Transformer):
    """Fill missing values in target columns using simple methods.

    This transformer handles missing values (NaN) in target columns using
    either a fixed value or backfill/forward fill methods. It operates
    on columns identified by semantic roles.

    Args:
        target_roles: Role(s) identifying columns to fill. Defaults to FeatureRole().
        values: Value or dictionary of column-specific values to use for filling.
            Defaults to None (no filling).
        method: Filling method - 'bfill' for backward fill or 'ffill' for forward fill.
            Defaults to None (no filling).
        key: Optional executor identifier.

    Example:
        >>> from hypex.transformers import NaFiller
        >>> from hypex.dataset import FeatureRole
        >>> filler = NaFiller(target_roles=FeatureRole(), values=0)
        >>> result = filler.execute(experiment_data)
    """

    def __init__(
        self,
        target_roles: str | Sequence[str] | None = None,
        values: ScalarType | dict[str, ScalarType] | None = None,
        method: Literal["bfill", "ffill"] | None = None,
        key: Any = "",
    ):
        super().__init__(key=key)
        self.target_roles = target_roles or FeatureRole()
        self.values = values
        self.method = method

    @staticmethod
    def _inner_function(
        data: Dataset,
        target_cols: str | None = None,
        values: ScalarType | dict[str, ScalarType] | None = None,
        method: Literal["bfill", "ffill"] | None = None,
    ) -> Dataset:
        target_cols = Adapter.to_list(target_cols)
        for column in target_cols:
            value = values[column] if isinstance(values, dict) else values
            data[column] = data[column].fillna(values=value, method=method)
        return data

    def execute(self, data: ExperimentData) -> ExperimentData:
        target_cols = data.ds.search_columns(roles=self.target_roles)
        result = data.copy(
            data=self.calc(
                data=data.ds,
                target_cols=target_cols,
                values=self.values,
                method=self.method,
            )
        )
        return result


class MLNaFilter(MLtransformer):
    """Fill missing values using statistical strategies learned from data.

    This transformer learns statistics (mean, median, mode, or zero) from
    target columns during fitting and uses them to fill missing values
    during prediction. The learned statistics are persisted for model
    reproducibility.

    Supported strategies:
        - 'mean': Fill with column mean
        - 'median': Fill with column median (default)
        - 'mode': Fill with column mode
        - 'zero': Fill with zero

    Args:
        strategy: Statistical strategy to use for filling missing values.
            Defaults to 'median'.
        target_roles: Role(s) identifying columns to fill. Defaults to None.
        stats: Pre-computed statistics dictionary. Defaults to None.
        key: Optional executor identifier.
        **calc_kwargs: Additional keyword arguments for the calc method.

    Example:
        >>> from hypex.transformers import MLNaFilter
        >>> from hypex.dataset import FeatureRole
        >>> filler = MLNaFilter(strategy='median', target_roles=FeatureRole())
        >>> filler.fit(experiment_data)
        >>> result = filler.predict(experiment_data)
        >>> # Or use through execute:
        >>> result = filler.execute(experiment_data)
    """

    def __init__(
        self,
        strategy: Literal["mean", "median", "mode", "zero"] = "median",
        target_roles: ABCRole | list[ABCRole] | None = None,
        stats: dict[str, Any] | None = None,
        key = "",
        **calc_kwargs
    ):
        super().__init__(target_roles, key, **calc_kwargs)

        self.strategy = strategy
        self._model = self._validate_stats(stats)

    @staticmethod
    def _validate_stats(stats: dict[str, Any] | None) -> dict[str, Any]:
        if stats is None:
            return {}

        for col in stats:
            if (
                isinstance(stats[col], int) or
                isinstance(stats[col], float)
            ):
                continue
            else:
                raise ValueError(f"{col} stats has incorrect value.")
        return stats

    # ── fit ─────────────────────────────────

    def fit(self, data: ExperimentData):
        """Learn statistics from target columns for filling missing values.

        This method can be used standalone or through ``execute``. When
        called standalone, it returns the fitted data. When used through
        ``execute``, it sets the ``is_fitted`` flag and returns the fitted
        data for subsequent operations.

        Args:
            data: The experiment data container.

        Returns:
            The fitted experiment data.

        Example:
            >>> filler = MLNaFilter(strategy='median')
            >>> fitted = filler.fit(experiment_data)
        """
        columns = data.field_search(
            roles=self.target_roles,
            search_types=[int, float],
        )

        for col in columns:
            col_data = data.ds[col]
            if self.strategy == "mean":
                self._model[col] = float(col_data.mean())
            elif self.strategy == "median":
                self._model[col] = float(col_data.quantile(0.5))
            elif self.strategy == "mode":
                self._model[col] = col_data.mode().get_values(0, col)
            elif self.strategy == "zero":
                self._model[col] = 0
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

        self.is_fitted = True
        return data

    # ── predict ─────────────────────────────────

    def predict(self, data: ExperimentData):
        """Fill missing values using learned statistics.

        This method can be used standalone or through ``execute``. When
        called standalone, it returns the data with filled values. When
        used through ``execute``, it applies the filling to the dataset.

        Args:
            data: The experiment data container.

        Returns:
            The experiment data with filled missing values.

        Example:
            >>> filler = MLNaFilter(strategy='median')
            >>> fitted = filler.fit(experiment_data)
            >>> filled = filler.predict(experiment_data)
        """
        if not self._model:
            raise ValueError(
                "MLNaFilter is not fitted. "
                "Run in FIT or FIT_PREDICT mode first, or load an artifact."
            )
        for col, fill_value in self._model.items():
            if col in data.ds.columns:
                filled = data.ds[col].fillna(values=fill_value)
                data.ds[col] = filled
        return data

    # ── serialization ─────────────────────────────────

    def _save_model_backend(self, model: dict[str, Any], path: str, postfix: str):
        """Save learned statistics to JSON file.

        Args:
            model: Dictionary of column statistics.
            path: Directory path to save the model.
            postfix: Postfix for the model filename.
        """
        if not model:
            raise ValueError(
                "MLNaFilter is not fitted. "
                "Run in FIT or FIT_PREDICT mode first, or load an artifact."
            )

        model_name = self.id + "_" + postfix
        model_path = os.path.join(path, f"{model_name}.json")
        with open(model_path, "w") as f:
            json.dump(model, f, indent=4)

    def _load_model_backend(self, path: str, postfix: str):
        """Load learned statistics from JSON file.

        Args:
            path: Directory path containing the model file.
            postfix: Postfix for the model filename.

        Returns:
            Dictionary of loaded column statistics.
        """
        model_name = self.id + "_" + postfix
        model_path = os.path.join(path, f"{model_name}.json")
        if not os.path.isfile(model_path):
            raise FileExistsError(f"No file found for {model_name} executor.")
        with open(model_path) as f:
            model = json.load(f)
        self._validate_stats(model)
        return model
