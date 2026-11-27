from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Literal

from ..dataset import ABCRole, Dataset, ExperimentData
from .abstract import MLtransformer


class MLScaler(MLtransformer):
    """Scale numeric features using statistical transformations.

    This transformer scales target columns using either standard scaling
    (z-score) or min-max normalization. The scaling parameters are learned
    during fitting and persisted for model reproducibility.

    Supported strategies:
        - 'standart': Standard scaling (z-score) with mean=0 and std=1
        - 'min-max': Min-max normalization to range [0, 1]

    Args:
        strategy: Scaling strategy to use. Defaults to 'standart'.
        target_roles: Role(s) identifying columns to scale. Defaults to None.
        key: Optional executor identifier.
        **kwargs: Additional keyword arguments for the calc method.

    Example:
        >>> from hypex.transformers import MLScaler
        >>> from hypex.dataset import FeatureRole
        >>> scaler = MLScaler(strategy='standart', target_roles=FeatureRole())
        >>> scaler.fit(experiment_data)
        >>> result = scaler.predict(experiment_data)
        >>> # Or use through execute:
        >>> result = scaler.execute(experiment_data)
    """

    def __init__(
        self,
        strategy: Literal["standart", "min-max"] = "standart",
        target_roles: ABCRole | list[ABCRole] | None = None,
        key="",
        **kwargs
    ):
        super().__init__(target_roles, key, **kwargs)
        self.strategy = strategy
        self._model: defaultdict = defaultdict(dict)
        self._eps: float = 1e-8

    # ── fit ─────────────────────────────────

    def _minmax_fit(self, data: Dataset, columns: str | list[str]) -> None:
        """Learn min-max scaling parameters.

        Args:
            data: Input dataset.
            columns: Columns to scale.
        """
        stats = (
            data[columns]
            .agg(["min", "max"])
            .to_small_dataset()
            .to_dict()
            ["data"]
        )
        for key in stats["data"]:
            for idx, index in enumerate(stats["index"]):
                self._model[key][index] = stats["data"][key][idx]

    def _standart_fit(self, data: Dataset, columns: str | list[str]) -> Dataset:
        """Learn standard scaling parameters.

        Args:
            data: Input dataset.
            columns: Columns to scale.

        Returns:
            Dataset with mean and std statistics.
        """
        stats = (
            data[columns]
            .agg(["mean", "std"])
            .to_small_dataset()
            .to_dict()
            ["data"]
        )
        for key in stats["data"]:
            for idx, index in enumerate(stats["index"]):
                self._model[key][index] = stats["data"][key][idx]

    def fit(self, data: ExperimentData):
        """Learn scaling parameters from target columns.

        This method can be used standalone or through ``execute``. When
        called standalone, it returns the fitted data. When used through
        ``execute``, it sets the ``is_fitted`` flag and returns the fitted
        data for subsequent operations.

        Args:
            data: The experiment data container.

        Returns:
            The fitted experiment data.

        Example:
            >>> scaler = MLScaler(strategy='standart')
            >>> fitted = scaler.fit(experiment_data)
        """
        columns = data.field_search(
            roles=self.target_roles,
            search_types=[int, float],
        )
        if self.strategy == "standart":
            func = self._standart_fit
        elif self.strategy == "min-max":
            func = self._minmax_fit
        else:
            raise ValueError("Incorrect strategy. Use 'standart' or 'min-max'.")

        func(data.ds, columns)
        return data

    # ── predict ─────────────────────────────────

    def _minmax_predict(self, data: ExperimentData, columns: str | list[str]) -> Dataset:
        """Apply min-max scaling to columns.

        Args:
            data: Input dataset.
            columns: Columns to scale.

        Returns:
            Scaled dataset.
        """
        for col in columns:
            if col not in self._model:
                raise KeyError(f"{col} wasn't in fit dataset.")
            data.ds[col] = (
                (data.ds[col] - self._model[col]["min"]) /
                (self._model[col]["max"] - self._model[col]["min"] + self._eps)
            )
        return data

    def _standart_predict(self, data: ExperimentData, columns: str | list[str]) -> Dataset:
        """Apply standard scaling to columns.

        Args:
            data: Input dataset.
            columns: Columns to scale.

        Returns:
            Scaled dataset.
        """
        for col in columns:
            if col not in self._model:
                raise KeyError(f"{col} wasn't in fit dataset.")
            data.ds[col] = (
                (data.ds[col] - self._model[col]["mean"]) /
                (self._model[col]["std"] + self._eps)
            )
        return data

    def predict(self, data: ExperimentData):
        """Apply learned scaling to target columns.

        This method can be used standalone or through ``execute``. When
        called standalone, it returns the data with scaled columns. When
        used through ``execute``, it applies the scaling to the dataset.

        Args:
            data: The experiment data container.

        Returns:
            The experiment data with scaled columns.

        Example:
            >>> scaler = MLScaler(strategy='standart')
            >>> fitted = scaler.fit(experiment_data)
            >>> scaled = scaler.predict(experiment_data)
        """
        columns = data.field_search(
            roles=self.target_roles,
            search_types=[int, float],
        )

        if self.strategy == "standart":
            func = self._standart_predict
        elif self.strategy == "min-max":
            func = self._minmax_predict
        else:
            raise ValueError("Incorrect strategy. Use 'standart' or 'min-max'.")

        return func(data, columns)

    # ── serialization ─────────────────────────────────

    def _save_model_backend(self, model: dict[str, Any], path: str, postfix: str):
        """Save scaling parameters to JSON file.

        Args:
            model: Dictionary of scaling parameters.
            path: Directory path to save the model.
            postfix: Postfix for the model filename.
        """
        if not model:
            raise ValueError(
                "MLScaler is not fitted. "
                "Run in FIT or FIT_PREDICT mode first, or load an artifact."
            )

        model_name = self.id + "_" + postfix
        model_path = os.path.join(path, f"{model_name}.json")
        with open(model_path, "w") as f:
            json.dump(model, f, indent=4)

    def _load_model_backend(self, path: str, postfix: str):
        """Load scaling parameters from JSON file.

        Args:
            path: Directory path containing the model file.
            postfix: Postfix for the model filename.

        Returns:
            Dictionary of loaded scaling parameters.
        """
        model_name = self.id + "_" + postfix
        model_path = os.path.join(path, f"{model_name}.json")
        if not os.path.isfile(model_path):
            raise FileExistsError(f"No file found for {model_name} executor.")
        with open(model_path) as f:
            model = json.load(f)
        return model
