from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore

from ..dataset import Dataset
from ..dataset.roles import FeatureRole
from .abstract import Extension


# class CholeskyExtension(Extension):
#     def _calc_pandas(self, data: Dataset, epsilon: float = 1e-3, **kwargs):
#         cov = data.data.to_numpy()
#         cov = cov + np.eye(cov.shape[0]) * epsilon
#         return self.result_to_dataset(
#             pd.DataFrame(np.linalg.cholesky(cov), columns=data.columns),
#             {column: FeatureRole() for column in data.columns},
#         )

class UniteCovExtension(Extension):

    def calc(
         self, data: Dataset, test_data: Dataset | None = None   
    ):
        cov_data = data.data.cov().to_numpy()
        if test_data is None:
            cov_test = test_data.data.cov().to_numpy()
            result = (cov_data + cov_test) / 2
        else:
            result = cov_data

        return self.result_to_dataset(
            pd.DataFrame(result, columns=data.columns),
            {column: FeatureRole() for column in data.columns},
        )

class CholeskyExtension(Extension):
    def calc(
        self, data: Dataset, epsilon: float = 1e-3
    ):
        """
        Args
        ----
            data: `Dataset`
                features covariance matrix;
            
            epsilon: `float`
                Correction to result matrix.By default is `1e-3`.
        
        """
        cov = data.data.to_numpy()
        cov = cov + np.eye(cov.shape[0]) * epsilon
        return self.result_to_dataset(
            pd.DataFrame(np.linalg.cholesky(cov), columns=data.columns),
            {column: FeatureRole() for column in data.columns},
        )

class InverseExtension(Extension):
    def calc(self, data: Dataset, **kwargs):
        """
        Calculate inverse matrix.

        Args
        ----
            data: `Dataset`
                input matrix.
        """
        return self.result_to_dataset(
            pd.DataFrame(np.linalg.inv(data.data.to_numpy()), columns=data.columns),
            {column: FeatureRole() for column in data.columns},
        )
