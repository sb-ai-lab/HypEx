from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..dataset import ABCRole, AdditionalTargetRole, ExperimentData
from ..executor import ExecutorSnapshot, MLExecutor
from ..utils import ExperimentDataEnum
from .models import LinearModel


class LinearModelExecutor(MLExecutor):
    """Fit a LinearModel and add its predictions to the experiment data."""

    def __init__(
        self,
        model_params: dict[str, Any] | Sequence[tuple[str, Any]] | None = None,
        target_roles: ABCRole | None = None,
        feature_roles: ABCRole | list[ABCRole] | None = None,
        key: str = "",
    ) -> None:
        super().__init__(key=key)
        self.model_params = dict(model_params or {})
        self.target_roles = target_roles
        self.feature_roles = feature_roles

    @property
    def _is_transformer(self) -> bool:
        return True

    def fit(self, data: ExperimentData) -> ExperimentData:
        model = LinearModel(
            target_roles=self.target_roles,
            feature_roles=self.feature_roles,
            **self.model_params,
        )
        self.model = model.fit(data.ds)
        self.is_fitted = True
        return data

    def predict(self, data: ExperimentData) -> ExperimentData:
        if not self.has_model:
            raise ValueError("Fit or load LinearModelExecutor before calling predict.")
        prediction = self.model.predict(data.ds)
        return data.set_value(
            ExperimentDataEnum.additional_fields,
            prediction.columns[0],
            prediction,
            role=AdditionalTargetRole(float),
        )

    def get_artifact(self) -> ExecutorSnapshot:
        snapshot = super().get_artifact()
        # MLArtifact interprets dictionary parameters as role descriptors.
        snapshot.params["model_params"] = list(self.model_params.items())
        return snapshot

    def _model_path(self, path: str, postfix: str) -> str:
        return str(Path(path) / f"{self.id}_{postfix}.pkl")

    def _save_model_backend(self, model: LinearModel, path: str, postfix: str) -> None:
        model.save(self._model_path(path, postfix))

    def _load_model_backend(self, path: str, postfix: str) -> LinearModel:
        return LinearModel.load(self._model_path(path, postfix))

    def load_model(self, path: str, postfix: str = "") -> None:
        super().load_model(path, postfix)
        self.is_fitted = True
