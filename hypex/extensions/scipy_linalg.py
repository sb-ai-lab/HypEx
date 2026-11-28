from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore

from .._spark import Correlation, F, LinearRegression, VectorAssembler
from ..dataset import Dataset
from ..dataset.backends import PandasDataset, SparkDataset
from ..dataset.roles import FeatureRole, TargetRole
from ..utils.registry import backend_factory
from .abstract import Extension


class UniteCovExtension(Extension):
    @staticmethod
    def _validate_group_size(size: int) -> None:
        if size < 2:
            raise ValueError("Each group must have at least two complete rows.")

    @staticmethod
    def _calc_group(data: Dataset, columns: list[str]) -> tuple[np.ndarray, int]:
        implementation = backend_factory.resolve_backend(UniteCovExtension, data)
        if implementation is None:
            raise ValueError(f"No covariance implementation for {data.backend_type!r}.")
        return implementation._calc_single_data(data, columns)

    def calc(self, data: Dataset, test_data: Dataset | None = None, **kwargs) -> Dataset:
        """Compute sample covariance using complete rows within each group."""
        columns = data.columns
        if not columns:
            raise ValueError("At least one covariance column is required.")
        if test_data is not None and set(columns) - set(test_data.columns):
            raise ValueError("Incorrect columns sets of two input datasets.")

        result, size = self._calc_group(data, columns)
        if test_data is not None:
            test_cov, test_size = self._calc_group(test_data, columns)
            result = ((size - 1) * result + (test_size - 1) * test_cov) / (
                size + test_size - 2
            )

        return self.result_to_dataset(
            pd.DataFrame(result, index=columns, columns=columns),
            {column: FeatureRole() for column in columns},
        )


@backend_factory.register(UniteCovExtension, PandasDataset)
class PandasUniteCovExtension(UniteCovExtension):
    @classmethod
    def _calc_single_data(
        cls, data: Dataset, columns: list[str]
    ) -> tuple[np.ndarray, int]:
        frame = data.raw_data.loc[:, columns].dropna()
        size = len(frame)
        cls._validate_group_size(size)
        return frame.cov(ddof=1).to_numpy(), size


@backend_factory.register(UniteCovExtension, SparkDataset)
class SparkUniteCovExtension(UniteCovExtension):
    @classmethod
    def _calc_single_data(
        cls, data: Dataset, columns: list[str]
    ) -> tuple[np.ndarray, int]:
        frame = data.raw_data[columns].to_spark().na.drop()
        stats = frame.agg(
            F.count("*"), *[F.stddev_samp(column) for column in columns]
        ).first()
        size = stats[0]
        cls._validate_group_size(size)
        std = np.asarray(stats[1:], dtype=float)

        assembler = VectorAssembler(inputCols=columns)
        vector_column = assembler.getOutputCol()
        vectors = assembler.transform(frame).select(vector_column)
        matrix = Correlation.corr(vectors, vector_column, method="pearson").first()[0]
        covariance = matrix.toArray() * std[:, None]
        covariance *= std[None, :]
        # Spark Pearson treats variances <= 1e-12 as zero.
        small = std <= 1e-6
        covariance[small, :] = 0
        covariance[:, small] = 0
        np.fill_diagonal(covariance, std**2)
        return covariance, size


class CholeskyExtension(Extension):
    def calc(self, data: Dataset, epsilon: float = 1e-8):
        """
        Args
        ----
            data: `Dataset`
                features covariance matrix;

            epsilon: `float`
                Correction to result matrix.By default is `1e-8`.

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
