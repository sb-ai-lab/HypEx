from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from ..ml import CUPACExecutor, ModelSelectionExecutor
from ..splitters import CUPACDataSplitter
from ..utils.enums import MLModeEnum
from .ml import MLExperiment


class CupacExperiment(MLExperiment):
    """Dedicated MLExperiment for CUPAC pipeline orchestration."""

    def __init__(
        self,
        cupac_models: Union[str, Sequence[str], None] = None,
        mode: str | MLModeEnum = MLModeEnum.FIT_PREDICT,
        experiment_id: Optional[str] = None,
        key: Any = "",
        n_folds: int = 5,
        random_state: Optional[int] = None,
        cv_aggregation: str = "mean",
        transformer: bool | None = None,
    ):
        selector_executor = ModelSelectionExecutor(
            models=cupac_models,
            n_folds=n_folds,
            random_state=random_state,
            cv_aggregation=cv_aggregation,
        )
        cupac_executor = CUPACExecutor(
            cupac_models=cupac_models,
            n_folds=n_folds,
            random_state=random_state,
            cv_aggregation=cv_aggregation,
        )

        super().__init__(
            splitters=[CUPACDataSplitter()],
            transformers=[],
            ml_executors=[selector_executor, cupac_executor],
            mode=mode,
            experiment_id=experiment_id,
            transformer=transformer,
            key=key,
        )
