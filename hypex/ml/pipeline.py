from __future__ import annotations

from typing import Any, Sequence

from ..dataset import ExperimentData
from ..dataset.ml_data import MLExperimentData
from ..executor import Executor
from ..utils import PredictorModeEnum, TransformerModeEnum
from .base import MLExecutor, MLPredictor, MLTransformer


class MLExperiment(Executor):
    """Sequential ML experiment composed of MLExecutor steps.

    Executes steps in order, threading MLExperimentData through each.
    Supports switching all steps to predict/transform mode for inference.
    """

    def __init__(self, steps: Sequence[Executor], key: Any = ""):
        """Initialise the experiment.

        Args:
            steps: Ordered sequence of executors to apply.
                Typically MLTransformers, MLPredictors, or other
                Experiment subclasses (e.g. ModelSelection).
            key: Optional executor key.
        """
        self.steps = list(steps)
        super().__init__(key=key)

    def execute(self, data: ExperimentData) -> MLExperimentData:
        """Run all steps sequentially, threading data through each.

        Args:
            data: Input experiment data.

        Returns:
            MLExperimentData produced by the last step.
        """
        result: ExperimentData = data
        for step in self.steps:
            result = step.execute(result)
        return result  # type: ignore[return-value]

    def set_predict_mode(self) -> None:
        """Switch all steps to predict/transform mode for inference."""
        for step in self.steps:
            if isinstance(step, MLTransformer):
                step.mode = TransformerModeEnum.transform
            elif isinstance(step, MLPredictor):
                step.mode = PredictorModeEnum.predict

    def set_fit_mode(self) -> None:
        """Switch all steps back to fit_transform/fit_predict mode for training."""
        for step in self.steps:
            if isinstance(step, MLTransformer):
                step.mode = TransformerModeEnum.fit_transform
            elif isinstance(step, MLPredictor):
                step.mode = PredictorModeEnum.fit_predict
