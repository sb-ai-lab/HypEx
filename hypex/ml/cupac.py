from typing import Any, Optional
import numpy as np
from copy import deepcopy
from ..dataset.dataset import Dataset, ExperimentData
from ..dataset.roles import TargetRole, StatisticRole
from ..executor import MLExecutor
from ..utils import ExperimentDataEnum
from ..utils.enums import BackendsEnum


from ..extensions.cupac import CupacExtension

from typing import Union, Sequence
from ..utils.models import CUPAC_MODELS


class CUPACExecutor(MLExecutor):
    """Executor that fits predictive models to pre-period covariates and adjusts target
    features using the CUPAC approach (model-based prediction adjustment similar to CUPED).
    """

    def __init__(
        self,
        cupac_features: dict[str, list],
        cupac_model: Union[str, Sequence[str], None] = None,
        key: Any = "",
        n_folds: int = 5,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            cupac_features: dict[str, list] — parameters for CUPAC, e.g. {"target": ["cov1", "cov2"]}
            cupac_model: str or list of str — model name (e.g. 'linear', 'ridge', 'lasso', 'catboost') or list of model names to try.
            key: key for executor.
            n_folds: number of folds for cross-validation.
            random_state: random seed.
        """
        super().__init__(target_role=TargetRole(), key=key)
        self.cupac_features = cupac_features
        self.cupac_model = cupac_model
        self.n_folds = n_folds
        self.random_state = random_state
        self.fitted_models: dict[str, Any] = {}
        self.best_model_names: dict[str, str] = {}
        self.is_fitted = False

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        cupac_features: dict[str, list],
        n_folds: int = 5,
        random_state: Optional[int] = None,
        cupac_model: Optional[str] = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        instance = cls(
            cupac_features=cupac_features,
            n_folds=n_folds,
            random_state=random_state,
            cupac_model=cupac_model,
        )
        instance.fit(data)
        return instance.predict(data)

    def get_models(self, backend_name: str) -> tuple[dict[str, Any], Sequence[str]]:
        """
        Get available models for backend and select explicit models to use.
        Returns (available_models_dict, explicit_models_list)
        """
        # Get models available for this backend
        available_models = {}
        for model_name, backends in CUPAC_MODELS.items():
            if backend_name in backends and backends[backend_name] is not None:
                available_models[model_name.lower()] = backends[backend_name]

        # Select explicit models
        if self.cupac_model:
            if isinstance(self.cupac_model, str):
                names = [self.cupac_model.lower()]
            else:
                names = [m.lower() for m in self.cupac_model]

            # Filter to only models available for current backend
            available_names = [name for name in names if name in available_models]
            return available_models, available_names
        return available_models, list(available_models.keys())

    def fit(self, X: Dataset) -> "CUPACExecutor":
        # Backend is guaranteed to be correct by dataset creation
        backend_name = X.backend.name

        available_models, explicit_models = self.get_models(backend_name)

        self.extension = CupacExtension(
            cupac_features=self.cupac_features,
            available_models=available_models,
            explicit_models=explicit_models,
            n_folds=self.n_folds,
            random_state=self.random_state,
        )
        self.extension.calc(X, mode="fit")
        self.is_fitted = True
        return self

    def predict(self, X: Dataset) -> dict[str, np.ndarray]:
        if not hasattr(self, "extension"):
            raise RuntimeError("CUPACExecutor not fitted. Call fit() first.")
        return self.extension.calc(X, mode="predict")

    def get_variance_reductions(self):
        if not hasattr(self, "extension"):
            raise RuntimeError("CUPACExecutor not fitted. Call fit() first.")
        return self.extension.get_variance_reductions()

    def execute(self, data: ExperimentData) -> ExperimentData:
        self.fit(data.ds)
        predictions = self.predict(data.ds)
        new_ds = deepcopy(
            data.ds
        )  # Create a deep copy to avoid modifying original dataset
        for key, pred in predictions.items():
            if hasattr(pred, "values"):
                pred = pred.values
            new_ds = new_ds.add_column(data=pred, role={key: TargetRole()})
        # Save variance reductions to additional_fields
        variance_reductions = self.get_variance_reductions()
        for key, reduction in variance_reductions.items():
            data.additional_fields = data.additional_fields.add_column(
                data=[reduction], role={key: StatisticRole()}
            )
        return data.copy(data=new_ds)
