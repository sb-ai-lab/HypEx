from __future__ import annotations

import numpy as np
import pyspark.sql.functions as F

from pyspark.sql import DataFrame as SparkDF
from pyspark import StorageLevel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

from .abstract import Extension

from ..dataset import (
    Dataset, 
    ExperimentData, 
    ABCRole,
    AdditionalMatchingRole,
    FeatureRole,
    TargetRole,
    InfoRole
)
from ..dataset.backends import PandasDataset, SparkDataset
from ..utils.registry import backend_factory
from ..utils import Adapter

class BiasExtension(Extension):
    def __init__(
            self,
            grouping_role: ABCRole,
            target_roles: list[ABCRole],
    ):
        super().__init__()
        self.grouping_role = grouping_role
        self.target_roles = target_roles

        self.target_field = None
        self.group_field = None
        self.features = None

    def _set_columns(self, data: Dataset) -> list[str]:
        self.target_field = data.search_columns(self.target_roles)[0]
        self.group_field = data.search_columns(self.grouping_role)[0]

    @staticmethod
    def prepare_data(
        data: ExperimentData
    ) -> Dataset:
        raise NotImplementedError

    @staticmethod
    def calc_bias(
            X: Dataset, X_matched: Dataset, coefficients: np.ndarray[float]
    ):
        raise NotImplementedError
    
    def calc(self, data: Dataset, **kwargs):
        raise NotImplementedError
    
    def _calc_coefs(self, data: Dataset) -> np.ndarray:
        raise NotImplementedError
    
    @staticmethod
    def _extract_info(data: Dataset) -> tuple[Dataset, list[str], list[str]]:
        neighbors_cols = data.search_columns(AdditionalMatchingRole())
        if len(neighbors_cols) == 0:
            raise ValueError("No indexes were found")
        
        numeric_cols = data.search_columns(
            roles=[
                FeatureRole(), TargetRole(), 
            ], 
            search_types=[int, float]
        )
        return neighbors_cols, numeric_cols

# TODO: pandas approach should be realized sonner
@backend_factory.register(BiasExtension, PandasDataset)
class PandasBisaExtesion(BiasExtension):
    @staticmethod
    def _prepare_data(
        data: Dataset, 
        neighbors_cols: list[str] | str, 
        numeric_cols: list[str] | str
    ):
        # Nothing changes in pandas approach 
        # TODO: realize in 'clear' pandas
        neighbors_cols = Adapter.to_list(neighbors_cols)
        numeric_cols = Adapter.to_list(numeric_cols)
        
        t_data = data[numeric_cols]
        indexes = data[neighbors_cols]
        # additional fields are already allignet according to index

        matched_data = (
            indexes
            .apply(list, axis=1, role={'_index' : InfoRole()})
            .explode('_index')
            .merge(t_data, 
                   left_on='_index' ,right_index=True)
            .drop(columns=['_index'])
            .reset_index()
            .groupby(by='index')
            .agg('mean')
            .rename({col: col + "_matched" for col in numeric_cols})
        )
        matched_data.index.name = None

        return indexes, matched_data

@backend_factory.register(BiasExtension, SparkDataset)
class SparkBisaExtesion(BiasExtension):
    PERSIST_POLITIC = StorageLevel.MEMORY_AND_DISK

    @staticmethod
    def _prepare_data(
        data: Dataset, 
        neighbors_cols: list[str] | str, 
        numeric_cols: list[str] | str
    ) -> SparkDF:
        neighbors_cols = Adapter.to_list(neighbors_cols)
        numeric_cols = Adapter.to_list(numeric_cols)
        
        t_data: SparkDF = data[numeric_cols].data.to_spark(index_col='index')
        indexes: SparkDF = data[neighbors_cols].data.to_spark(index_col='index')
        working_columns = [col for col in indexes.columns if col != 'index']

        matched_data = (
            indexes.select(
                F.col('index').alias('initial_index'),
                F.explode(F.array(*working_columns).alias("list_indexes")).alias('index')
            )
            .join(other=t_data, on='index')
            .groupBy('initial_index')
            .agg(
                *[
                    F.mean(col).alias(col + "_matched") for col in t_data.columns if col != 'index'
                ]
            )
        )

        return matched_data
    
    @classmethod
    def prepare_data(cls, data: Dataset) -> tuple[Dataset]:
        neighbors_cols, numeric_cols = cls._extract_info(data)
        matched_data = cls._prepare_data(
            data = data,
            neighbors_cols=neighbors_cols,
            numeric_cols=numeric_cols
        )
        matched_data = cls.result_to_dataset(matched_data, small=False)
        matched_data = matched_data.set_index('initial_index')
        matched_data.index.name = None
        
        indexes = data[neighbors_cols]

        return indexes, matched_data
    
    def _calc_coefs(self, data: SparkDF) -> np.ndarray:
        group_1, group_2, *_ = map(
            lambda row: row[0], 
            data.select(self.group_field).distinct().collect()
        )
        features = [col + "_matched" for col in self.features]
        asembler = VectorAssembler(inputCols=features, outputCol='_features')
        transformed_data = asembler.transform(data)
        lr = LinearRegression(
            featuresCol='_features', 
            labelCol=self.target_field + "_matched", 
            regParam=0.01
        )

        fit_data_1 = transformed_data.filter(F.col(self.group_field) == group_1)
        fit_data_2 = transformed_data.filter(F.col(self.group_field) == group_2)
        # TODO: Are these persists nessesary?
        fit_data_1.persist()
        fit_data_2.persist()

        model_1 = lr.fit(fit_data_1)
        model_2 = lr.fit(fit_data_2)

        fit_data_1.unpersist()
        fit_data_2.unpersist()

        weights_1 = model_1.coefficients.toArray()
        weights_2 = model_2.coefficients.toArray()

        return np.array(
            [
                [*weights_1],
                [*weights_2]
            ]
        )

    def _calc_bias(
            self, 
            data: SparkDF, 
            coefficients_1: np.ndarray, 
            coefficients_2: np.ndarray
    ) -> SparkDF:
        group_1, group_2, *_ = map(
            lambda row: row[0], 
            data.select(self.group_field).distinct().collect()
        )
        initial_data = data.withColumn(
            "bias",
            F.when(
                F.col(self.group_field) == group_1,
                sum(
                    [
                        (F.col(col) - F.col(col + "_matched")) * coefficients_1[idx]
                        for idx, col in enumerate(self.features)
                    ]
                )
            )
            .when(
                F.col(self.group_field) == group_2,
                sum(
                    [
                        (F.col(col) - F.col(col + "_matched")) * coefficients_2[idx]
                        for idx, col in enumerate(self.features)
                    ]
                )
            )
            .otherwise(0)
        )

        final_data = initial_data.select(
            F.col('initial_index').alias("index"),
            F.col("bias"),
            F.col(self.target_field + "_matched").alias("matched_target")
        )
        return final_data
        
    def calc(self, data: Dataset, **kwargs) -> Dataset:
        storage_level = data.get_storage_level() or "MEMORY_AND_DISK"
        self._set_columns(data)
        neighbors_cols, numeric_cols = self._extract_info(data)
        self.features = [
                col for col in numeric_cols 
                if col != self.group_field and col != self.group_field
        ]

        matched_data = self._prepare_data(
            data = data,
            neighbors_cols=neighbors_cols,
            numeric_cols=numeric_cols
        )
        initial_data: SparkDF = data[numeric_cols + [self.group_field]].data.to_spark(index_col='initial_index')
        initial_data = initial_data.join(matched_data, on='initial_index')
        initial_data.persist(self.PERSIST_POLITIC)

        coefficients_1, coefficients_2  = self._calc_coefs(initial_data)
        final_data = self._calc_bias(initial_data, coefficients_1, coefficients_2)
        final_dataset: Dataset = self.result_to_dataset(final_data, {}, small=False).set_index("index")
        final_dataset.index.name = None

        final_dataset.persist(storage_level)
        initial_data.unpersist()

        return final_dataset