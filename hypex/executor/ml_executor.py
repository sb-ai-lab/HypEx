from __future__ import annotations

from abc import abstractmethod
from typing import Any, Literal

from ..dataset import Dataset, ExperimentData, FeatureRole
from ..utils import ExperimentDataEnum
from .executor import Executor


class MLExecutor(Executor):
    def __init__(
        self,
        model: Any | None = None,
        load_format: Literal["pickle", "joblib"] | None = None,
        mode: Literal["fit", "predict", "fit_predict"] = "fit_predict",
        key: Any = "",
    ):
        super().__init__(key=key)
        self.model = model
        self.load_format = load_format
        self.mode = mode
        self._fitted_model = None

    @abstractmethod
    def fit(self, X: Dataset, y: Dataset | None = None) -> MLExecutor:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Dataset) -> Dataset:
        raise NotImplementedError

    def score(self, X: Dataset, y: Dataset) -> float:
        raise NotImplementedError

    def save_state(self, path: str) -> None:
        import pickle

        try:
            import joblib
        except ImportError:
            joblib = None

        if self.load_format == "pickle":
            with open(path, "wb") as f:
                pickle.dump(self._fitted_model, f)
        elif self.load_format == "joblib":
            if joblib is None:
                raise ImportError("joblib is required for joblib format")
            joblib.dump(self._fitted_model, path)
        else:
            raise ValueError(f"Unsupported load_format: {self.load_format}")

    def load_state(self, path: str) -> None:
        import pickle

        try:
            import joblib
        except ImportError:
            joblib = None

        if self.load_format == "pickle":
            with open(path, "rb") as f:
                self._fitted_model = pickle.load(f)
        elif self.load_format == "joblib":
            if joblib is None:
                raise ImportError("joblib is required for joblib format")
            self._fitted_model = joblib.load(path)
        else:
            raise ValueError(f"Unsupported load_format: {self.load_format}")

    def _get_features_target(
        self, data: ExperimentData
    ) -> tuple[Dataset, Dataset | None]:
        features = data.ds.search_columns(FeatureRole())
        if not features:
            raise ValueError("No features found in dataset")

        X = data.ds[features]
        y = None

        return X, y

    def execute(self, data: ExperimentData) -> ExperimentData:
        X, y = self._get_features_target(data)

        if self.mode == "fit":
            self._fitted_model = self.fit(X, y)
        elif self.mode == "predict":
            if self._fitted_model is None:
                raise ValueError("Model must be fitted before prediction")
            predictions = self.predict(X)
            data = self._set_value(data, predictions)
        elif self.mode == "fit_predict":
            self._fitted_model = self.fit(X, y)
            predictions = self.predict(X)
            data = self._set_value(data, predictions)

        return data

    def _set_value(
        self, data: ExperimentData, value: Dataset, key: Any = None
    ) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.additional_fields, self.id, value, role=value.roles
        )
