from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..dataset import Dataset, ExperimentData, FeatureRole, PredictionRole, TargetRole
from ..dataset.ml_data import MLExperimentData
from ..executor import Executor
from ..utils import PredictorModeEnum, TransformerModeEnum


class MLExecutor(Executor, ABC):
    """Base class for ML executors with mode support.

    Unlike MatchingExecutor (used for CUPAC/Faiss matching),
    this class supports general-purpose ML pipelines with
    fit/transform/predict modes and artifact storage.
    """

    def __init__(self, key: Any = ""):
        super().__init__(key=key)

    def _ensure_ml_data(self, data: ExperimentData) -> MLExperimentData:
        """Wrap ExperimentData into MLExperimentData if needed."""
        if isinstance(data, MLExperimentData):
            return data
        return MLExperimentData.from_experiment_data(data)

    def _register(self, data: MLExperimentData, artifact: Any = None) -> None:
        """Register this executor in the pipeline and store artifact."""
        data.pipeline.append(self.id)
        if artifact is not None:
            data.artifacts[self.id] = artifact


class MLTransformer(MLExecutor, ABC):
    """ML executor for data transformations (scalers, encoders, etc.).

    Modes:
    - ``fit``: learn parameters, store as artifact
    - ``transform``: apply using stored artifact
    - ``fit_transform``: learn and apply
    """

    def __init__(
        self,
        mode: TransformerModeEnum = TransformerModeEnum.fit_transform,
        key: Any = "",
    ):
        super().__init__(key=key)
        self.mode = mode

    @abstractmethod
    def _fit(self, data: Dataset) -> Any:
        """Learn transformation parameters. Returns artifact."""

    @abstractmethod
    def _transform(self, data: Dataset, artifact: Any) -> Dataset:
        """Apply transformation using artifact. Returns transformed Dataset."""

    def execute(self, data: ExperimentData) -> MLExperimentData:
        """Execute the transformer according to its current mode.

        In ``fit`` / ``fit_transform`` mode the transformer learns
        parameters from the data and stores them as an artifact.
        In ``transform`` / ``fit_transform`` mode it applies the
        learned transformation to the dataset.

        Args:
            data: Input experiment data.

        Returns:
            MLExperimentData with the (optionally) transformed dataset.
        """
        ml_data = self._ensure_ml_data(data)

        if self.mode in (TransformerModeEnum.fit, TransformerModeEnum.fit_transform):
            artifact = self._fit(ml_data.ds)
            self._register(ml_data, artifact)
        else:
            artifact = ml_data.artifacts[self.id]

        if self.mode in (TransformerModeEnum.transform, TransformerModeEnum.fit_transform):
            ml_data._data = self._transform(ml_data.ds, artifact)

        return ml_data


class MLPredictor(MLExecutor, ABC):
    """ML executor for models (regression, classification, etc.).

    Modes:
    - ``fit``: train model, store as artifact
    - ``predict``: make predictions using stored artifact
    - ``fit_predict``: train and predict
    """

    def __init__(
        self,
        mode: PredictorModeEnum = PredictorModeEnum.fit_predict,
        key: Any = "",
    ):
        super().__init__(key=key)
        self.mode = mode

    @abstractmethod
    def _fit(self, features: Dataset, target: Dataset) -> Any:
        """Train model. Returns artifact (fitted model)."""

    @abstractmethod
    def _predict(self, features: Dataset, artifact: Any) -> Dataset:
        """Make predictions. Returns Dataset with predictions."""

    def _split_features_target(self, data: Dataset) -> tuple[Dataset, Dataset]:
        """Split Dataset into features and target by roles."""
        target_cols = data.search_columns(TargetRole())
        feature_cols = data.search_columns(FeatureRole())
        return data[feature_cols], data[target_cols]

    def execute(self, data: ExperimentData) -> MLExperimentData:
        """Execute the predictor according to its current mode.

        In ``fit`` / ``fit_predict`` mode the model is trained on
        features and target extracted from the dataset.
        In ``predict`` / ``fit_predict`` mode predictions are added
        to ``additional_fields`` with :class:`PredictionRole`.

        Args:
            data: Input experiment data.

        Returns:
            MLExperimentData with predictions in ``additional_fields``.
        """
        ml_data = self._ensure_ml_data(data)
        features, target = self._split_features_target(ml_data.ds)

        if self.mode in (PredictorModeEnum.fit, PredictorModeEnum.fit_predict):
            artifact = self._fit(features, target)
            self._register(ml_data, artifact)
        else:
            artifact = ml_data.artifacts[self.id]

        if self.mode in (PredictorModeEnum.predict, PredictorModeEnum.fit_predict):
            predictions = self._predict(features, artifact)
            for col in predictions.columns:
                ml_data.additional_fields = ml_data.additional_fields.add_column(
                    data=predictions[col],
                    role={f"{self.id}_{col}": PredictionRole()},
                )

        return ml_data
