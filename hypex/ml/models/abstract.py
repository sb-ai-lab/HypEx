from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ...dataset import ABCRole, Dataset, DatasetAdapter, FeatureRole, TargetRole
from ...utils import Adapter


class MLModel(ABC):
    def __init__(
        self,
        target_roles: ABCRole | None = None,
        feature_roles: ABCRole | list[ABCRole] | None = None,
        key: Any = "",
        **calc_kwargs,
    ):
        self.target_roles = target_roles or TargetRole()
        self.feature_roles = feature_roles or FeatureRole()
        self.key = key
        self.calc_kwargs = calc_kwargs

        self._model: Any = None
        self._features_dict: dict[str, ABCRole] = {}
        self._target_dict: dict[str, ABCRole] = {}

        self.is_fitted: bool = False

    @abstractmethod
    def _check_model_signature(self, model: Any):
        raise NotImplementedError

    @staticmethod
    def result_to_dataset(
        result: Any, roles: ABCRole | dict[str, ABCRole], small: bool = True
    ) -> Dataset:
        return DatasetAdapter.to_dataset(result, roles=roles, small=small)

    def _get_fields(self, data: Dataset) -> tuple[str, list[str]]:
        search_types = self.search_types
        targets = data.search_columns(self.target_roles)
        if len(targets) != 1:
            raise ValueError("Expected exactly one column matching target_roles.")
        target = targets[0]
        if search_types and data.roles[target].data_type not in search_types:
            raise ValueError(f"Unsupported target column type: {target!r}.")

        features = data.search_columns(
            Adapter.to_list(self.feature_roles), search_types=search_types
        )
        if not features or target in features:
            raise ValueError("Select feature columns that exclude the target.")
        return target, features

    # ── properties ─────────────────────────────────

    @property
    def search_types(self) -> list[type] | None:
        return [int, float]

    @property
    @abstractmethod
    def model_params(self) -> dict[str, Any]:
        raise NotImplementedError

    @model_params.setter
    def model_params(self, params: dict[str, Any]) -> None:
        self.set_model_params(params)

    @abstractmethod
    def set_model_params(self, dct: dict[str, Any]) -> None:
        """Validate and apply the supplied model parameter updates."""
        raise NotImplementedError

    @property
    def model(self) -> Any:
        return self._model

    @model.setter
    def model(self, value: Any) -> None:
        self._check_model_signature(value)
        self._model = value

    @property
    def fitted(self) -> bool:
        return self.is_fitted

    @property
    def model_stats(self) -> dict[str, Any]:
        return self._model_params

    @property
    def features_dict(self) -> dict[str, ABCRole]:
        return self._features_dict

    @property
    def target_dict(self) -> dict[str, ABCRole]:
        return self._target_dict

    # ── execution ─────────────────────────────────

    @abstractmethod
    def fit(self, data: Dataset) -> MLModel:
        """Train on a dataset containing feature and target columns."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Dataset) -> Dataset:
        raise NotImplementedError

    # ── MODEL serialization protocol ───────────────

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> MLModel:
        raise NotImplementedError
