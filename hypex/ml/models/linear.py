from __future__ import annotations

import os
import pickle
from copy import deepcopy
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from ...dataset import ABCRole, Dataset
from ...utils.registry import backend_factory
from .abstract import MLModel


def _l1_ratio(elastic_net_param: float, reg_type: Literal["l1", "l2"]) -> float:
    """Interpret zero as a request to use the penalty selected by reg_type."""
    return elastic_net_param or float(reg_type == "l1")


class LinearModel(MLModel):
    """Linear regression with optional L1, L2 or elastic net regularization.

    A zero ``reg_param`` selects ordinary least squares. For positive
    ``reg_param``, a positive ``elastic_net_param`` sets the L1 share;
    zero uses ``reg_type``. Regularization strength follows the native
    estimator's convention. Solver names and applicable convergence
    parameters depend on the estimator.

    Training resolves the feature and target columns from the dataset's roles.
    Predictions retain the input index and use the name ``<target>_predicted``.
    """

    def __init__(
        self,
        target_roles: ABCRole | None = None,
        feature_roles: ABCRole | list[ABCRole] | None = None,
        fit_intercept: bool = True,
        reg_param: float = 0.0,
        elastic_net_param: float = 0.0,
        reg_type: Literal["l1", "l2"] = "l1",
        solver: str | None = None,
        max_iter: int = 1000,
        tol: float = 0.0001,
        key: Any = "",
        **kwargs,
    ) -> None:
        super().__init__(target_roles, feature_roles, key, **kwargs)
        self.fit_intercept = fit_intercept
        self.reg_param = reg_param
        self.elastic_net_param = elastic_net_param
        self.reg_type = reg_type
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol

    @MLModel.model_params.getter
    def model_params(self) -> dict[str, Any]:
        return {
            "fit_intercept": self.fit_intercept,
            "reg_param": self.reg_param,
            "elastic_net_param": self.elastic_net_param,
            "reg_type": self.reg_type,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    def set_model_params(self, dct: dict[str, Any]) -> None:
        unknown = dct.keys() - self.model_params.keys()
        if unknown:
            raise ValueError(f"Unknown linear model parameters: {sorted(unknown)}")
        for key, value in dct.items():
            setattr(self, key, value)

    @property
    def search_types(self) -> list[type]:
        return [int, float]

    def _init_params(self) -> dict[str, Any]:
        return {
            **self.calc_kwargs,
            **self.model_params,
            "target_roles": self.target_roles,
            "feature_roles": self.feature_roles,
            "key": self.key,
        }

    # ── Backend selection ────────────────────────────────────────────

    def _check_model_signature(self, model: Any) -> None:
        if not isinstance(model, LinearModel):
            raise TypeError("Expected a linear model backend.")

    @staticmethod
    def _resolve_backend(data: Dataset) -> type[LinearModel]:
        backend_cls = backend_factory.resolve_backend(LinearModel, data)
        if backend_cls is None:
            raise ValueError(
                f"No linear regression backend registered for {data.backend_type}."
            )
        return backend_cls

    # ── Training ────────────────────────────────────────────

    def fit(self, data: Dataset) -> LinearModel:
        backend_cls = self._resolve_backend(data)
        if not isfinite(self.reg_param) or self.reg_param < 0:
            raise ValueError("reg_param must be finite and non-negative.")
        if not 0 <= self.elastic_net_param <= 1:
            raise ValueError("elastic_net_param must be between 0 and 1.")
        if self.reg_type not in ("l1", "l2"):
            raise ValueError("reg_type must be 'l1' or 'l2'.")

        target, features = self._get_fields(data)
        feature_roles = {name: deepcopy(data.roles[name]) for name in features}
        target_roles = {f"{target}_predicted": data.roles[target].astype(float)}
        backend = backend_cls(**self._init_params())
        backend._fit_native(data, features, target)

        self.model = backend
        self._features_dict = feature_roles
        self._target_dict = target_roles
        self.is_fitted = True
        return self

    def _fit_native(self, data: Dataset, features: list[str], target: str) -> None:
        raise NotImplementedError

    # ── Prediction ────────────────────────────────────────────

    def predict(self, X: Dataset) -> Dataset:
        if not self.fitted:
            raise ValueError("Fit or load LinearModel before calling predict.")
        backend = self.model
        if not isinstance(backend, self._resolve_backend(X)):
            # TODO: add convertion from one backend to another
            raise ValueError("Prediction data must use the fitted model's backend.")
        features = list(self._features_dict)
        invalid = [
            name
            for name in features
            if name not in X.roles or X.roles[name].data_type not in self.search_types
        ]
        if invalid:
            raise ValueError(f"Missing or non-numeric feature columns: {invalid}")
        output, role = next(iter(self._target_dict.items()))
        return backend._predict_native(X, features, output, deepcopy(role))

    def _predict_native(
        self, X: Dataset, features: list[str], output: str, role: ABCRole
    ) -> Dataset:
        raise NotImplementedError

    # ── Serialization ────────────────────────────────────────────

    def dumps(self) -> bytes:
        raise NotImplementedError

    @classmethod
    def loads(cls, data: bytes, **params: Any) -> LinearModel:
        """Restore the native estimator in a backend-specific instance."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save the HypEx state and native model bytes in one pickle file."""
        if not self.fitted:
            raise ValueError("Fit LinearModel before saving it.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        backend = self.model
        state = {
            "backend": type(backend),
            "params": self._init_params(),
            "features": self._features_dict,
            "target": self._target_dict,
        }
        artifact = {"state": state, "model": backend.dumps()}
        with TemporaryDirectory(dir=destination.parent) as temporary:
            staged_file = Path(temporary) / "model.pkl"
            with staged_file.open("wb") as stream:
                pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.link(staged_file, destination)

    @classmethod
    def load(cls, path: str) -> LinearModel:
        """Restore the wrapper state and native model from a single pickle file."""
        with Path(path).open("rb") as stream:
            artifact = pickle.load(stream)
        state = artifact["state"]
        backend_cls = state["backend"]
        if not isinstance(backend_cls, type) or not issubclass(
            backend_cls, LinearModel
        ):
            raise ValueError(f"Linear regression backend is unavailable: {backend_cls}")
        backend = backend_cls.loads(artifact["model"], **state["params"])
        instance = cls(**state["params"])
        instance.model = backend
        instance._features_dict = state["features"]
        instance._target_dict = state["target"]
        instance.is_fitted = True
        return instance
