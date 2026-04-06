from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence, Union

from ..ml import CUPACExecutor, ModelSelectionExecutor
from ..ml.feature_selection import FeatureSelectionExecutor
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
        feature_selection: Optional[
            Union[bool, Literal["importance", "variance", "correlation", "mutual_info"]]
        ] = None,
        feature_selection_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            cupac_models: Model(s) to try for CUPAC.
            mode: Execution mode ('fit', 'predict', 'fit_predict').
            experiment_id: ID to load models from (for predict mode).
            key: Experiment identifier.
            n_folds: Number of CV folds.
            random_state: Random seed.
            cv_aggregation: CV aggregation method.
            transformer: Whether to deepcopy data between executors.
            feature_selection: Feature selection method or True for default ("importance"):
                - None/False: No feature selection
                - True: Use "importance" method with default params
                - "importance": Select by model feature importance
                - "variance": Remove low-variance features
                - "correlation": Remove highly correlated features
                - "mutual_info": Select by mutual information
            feature_selection_params: Additional params for FeatureSelectionExecutor:
                - n_features (int|float): Number or fraction of features to keep
                - threshold (float): Score threshold for selection
                - importance_model (str): Model for importance method
        """
        # Build ML executors list
        ml_executors = []

        # Add feature selection if enabled
        if feature_selection:
            fs_params = feature_selection_params or {}
            method = (
                "importance" if feature_selection is True else feature_selection
            )
            fs_executor = FeatureSelectionExecutor(method=method, **fs_params)
            ml_executors.append(fs_executor)

        # Model selection
        selector_executor = ModelSelectionExecutor(
            models=cupac_models,
            n_folds=n_folds,
            random_state=random_state,
            cv_aggregation=cv_aggregation,
        )
        ml_executors.append(selector_executor)

        # CUPAC executor
        cupac_executor = CUPACExecutor(
            cupac_models=cupac_models,
            n_folds=n_folds,
            random_state=random_state,
            cv_aggregation=cv_aggregation,
        )
        ml_executors.append(cupac_executor)

        super().__init__(
            splitters=[CUPACDataSplitter()],
            transformers=[],
            ml_executors=ml_executors,
            mode=mode,
            experiment_id=experiment_id,
            transformer=transformer,
            key=key,
        )
