from __future__ import annotations

from typing import Any, Sequence

from ..dataset import Dataset, ExperimentData, FeatureRole, TargetRole
from ..dataset.ml_data import MLExperimentData
from ..experiments import Experiment
from ..extensions.model_selection import CrossValidationExtension
from ..utils import ExperimentDataEnum, PredictorModeEnum
from .metrics import MLMetric
from .predictors import SklearnPredictor


class ModelSelection(Experiment):
    """Cross-validates predictors and selects the best one.

    Orchestrates:
    1. Cross-validation of each predictor (via CrossValidationExtension)
    2. Best model selection by main_metric
    3. Delegates fit + predict to the best predictor via its ``execute()``
    4. Runs metric executors on the predictions

    Results are stored in ExperimentData.variables[self.id]:
    - "cv_results": raw CV results dict
    - "best_model": name of the best model

    Use ModelSelectionReporter to format results into a readable Dataset.

    Attributes:
        best_predictor_: best predictor after execute (None before).
        cv_results_: raw CV results dict (None before).
    """

    def __init__(
        self,
        predictors: Sequence[SklearnPredictor],
        metrics: Sequence[MLMetric],
        cv: int = 5,
        main_metric: str | None = None,
        key: Any = "",
    ):
        """Initialise the model selection experiment.

        Args:
            predictors: Predictor executors to compare.
            metrics: Metric executors for evaluation.
            cv: Number of cross-validation folds.
            main_metric: Metric name used to select the best model.
                Defaults to the first metric in *metrics*.
            key: Optional executor key.
        """
        self.predictors = list(predictors)
        self.metric_executors = list(metrics)
        self.cv = cv
        self.main_metric = main_metric or self.metric_executors[0].metric_name
        self.cv_extension = CrossValidationExtension()
        self.best_predictor_: SklearnPredictor | None = None
        self.cv_results_: dict[str, Any] | None = None
        super().__init__(
            executors=[*self.predictors, *self.metric_executors],
            transformer=False,
            key=key,
        )

    def _split_features_target(self, data: Dataset) -> tuple[Dataset, Dataset]:
        """Split *data* into features and target subsets by roles."""
        target_cols = data.search_columns(TargetRole())
        feature_cols = data.search_columns(FeatureRole())
        return data[feature_cols], data[target_cols]

    def _cross_validate(
        self, features: Dataset, target: Dataset,
    ) -> tuple[dict[str, Any], dict[str, SklearnPredictor]]:
        """Run cross-validation for all predictors.

        Returns:
            Tuple of (cv_results dict, predictor_map by name).
        """
        models: dict[str, Any] = {}
        predictor_map: dict[str, SklearnPredictor] = {}
        for i, pred in enumerate(self.predictors):
            name = f"{pred.__class__.__name__}_{i}"
            models[name] = pred.extension._model_proto
            predictor_map[name] = pred

        metric_names = [m.metric_name for m in self.metric_executors]
        cv_results = self.cv_extension.calc(
            data=features,
            target=target,
            models=models,
            cv=self.cv,
            metrics=metric_names,
        )
        return cv_results, predictor_map

    def _select_best(
        self,
        cv_results: dict[str, Any],
        predictor_map: dict[str, SklearnPredictor],
    ) -> tuple[str, SklearnPredictor]:
        """Pick the best predictor from CV results by main_metric.

        Returns:
            Tuple of (best model name, best predictor).
        """
        main_metric_obj = next(
            m for m in self.metric_executors if m.metric_name == self.main_metric
        )
        comparator = max if main_metric_obj.higher_is_better else min
        best_name = comparator(
            cv_results, key=lambda k: cv_results[k][self.main_metric]["mean"]
        )
        return best_name, predictor_map[best_name]

    def _evaluate(self, data: MLExperimentData) -> MLExperimentData:
        """Run all metric executors on predictions."""
        for metric in self.metric_executors:
            data = metric.execute(data)
        return data

    def execute(self, data: ExperimentData) -> MLExperimentData:
        """Run cross-validation, select the best model, and evaluate.

        Args:
            data: Input experiment data.

        Returns:
            MLExperimentData with predictions in ``additional_fields``
            and results in ``variables``.
        """
        if isinstance(data, MLExperimentData):
            ml_data = data
        else:
            ml_data = MLExperimentData.from_experiment_data(data)

        features, target = self._split_features_target(ml_data.ds)

        cv_results, predictor_map = self._cross_validate(features, target)
        self.cv_results_ = cv_results

        best_name, best_predictor = self._select_best(cv_results, predictor_map)
        self.best_predictor_ = best_predictor

        best_predictor.mode = PredictorModeEnum.fit_predict
        ml_data = best_predictor.execute(ml_data)

        ml_data = self._evaluate(ml_data)

        ml_data.set_value(
            ExperimentDataEnum.variables,
            self.id,
            {"cv_results": cv_results, "best_model": best_name},
        )
        return ml_data
