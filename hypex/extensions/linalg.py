import numpy as np
import pandas as pd  # type: ignore

from hypex.dataset import Dataset
from hypex.dataset.roles import FeatureRole
from .abstract import Extension


class CholeskyExtension(Extension):
    def _calc_pandas(self, data: Dataset, epsilon: float = 1e-3, **kwargs):
        cov = data.data.to_numpy()
        cov = cov + np.eye(cov.shape[0]) * epsilon
        return self.result_to_dataset(np.linalg.cholesky(cov), FeatureRole())


class InverseExtension(Extension):
    def _calc_pandas(self, data: Dataset, **kwargs):
        return self.result_to_dataset(
            np.linalg.inv(data.data.to_numpy()), FeatureRole()
        )
