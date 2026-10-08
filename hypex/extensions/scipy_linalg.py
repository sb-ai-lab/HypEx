from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

from ..dataset import Dataset
from ..dataset.backends import PandasDataset, SparkDataset
from ..dataset.roles import FeatureRole, TargetRole
from ..utils.registry import backend_factory
from .abstract import Extension


class UniteCovExtension(Extension):
    def calc(self, data: Dataset, test_data: Dataset | None = None, **kwargs):
        cov_data = data.raw_data.cov().to_numpy()
        if test_data is None:
            result = cov_data
        else:
            cov_test = test_data.raw_data.cov().to_numpy()
            result = (cov_data + cov_test) / 2

        return self.result_to_dataset(
            pd.DataFrame(result, columns=data.columns),
            {column: FeatureRole() for column in data.columns},
        )


class CholeskyExtension(Extension):
    def calc(self, data: Dataset, epsilon: float = 1e-3):
        """
        Args
        ----
            data: `Dataset`
                features covariance matrix;

            epsilon: `float`
                Correction to result matrix.By default is `1e-3`.

        """
        cov = data.raw_data.to_numpy()
        cov = cov + np.eye(cov.shape[0]) * epsilon
        return self.result_to_dataset(
            pd.DataFrame(np.linalg.cholesky(cov), columns=data.columns),
            {column: FeatureRole() for column in data.columns},
        )


class InverseExtension(Extension):
    def calc(self, data: Dataset, other: Dataset | None = None, **kwargs):
        """
        Calculate inverse matrix.

        Args
        ----
            data: `Dataset`
                input matrix.
        """
        return self.result_to_dataset(
            pd.DataFrame(np.linalg.inv(data.raw_data.to_numpy()), columns=data.columns),
            {column: FeatureRole() for column in data.columns},
        )


class LstsqExtension(Extension):
    """
    Master-backend class for lstsq extension.
    """

    @staticmethod
    def get_columns(data: Dataset) -> list[str]:
        """
        Get features and target columns.

        Return
        ------
            `list` where first item is target and other is features
        """
        target = data.search_columns(TargetRole())[0]
        features = [col for col in data.columns if col != target]

        return [target, *features]


@backend_factory.register(LstsqExtension, PandasDataset)
class PandasLstsqExtension(LstsqExtension):
    """
    Slave-backend class for lstsq extension.
    """

    def calc(self, data: Dataset, other: Dataset | None = None, **kwargs):
        target, *features = self.get_columns(data)
        features_data = data.select(features).raw_data.to_numpy()
        X = np.column_stack([np.ones(len(features_data)), features_data])
        return np.linalg.lstsq(X, data[target].raw_data.values, rcond=-1)[0][1:]


@backend_factory.register(LstsqExtension, SparkDataset)
class SparkLstsqExtension(LstsqExtension):
    """
    Slave-backend class for lstsq extension.
    """

    def calc(self, data: SparkDataset, other: Dataset | None = None, **kwargs):
        target, *features = self.get_columns(data)
        assembler = VectorAssembler(inputCols=features, outputCol="_features")

        transformed_data = assembler.transform(data.raw_data.to_spark()).select(
            target, "_features"
        )
        lr = LinearRegression(featuresCol="_features", labelCol=target, regParam=0.01)
        model = lr.fit(transformed_data)

        weights = model.coefficients.toArray().reshape(-1, 1)
        return weights
