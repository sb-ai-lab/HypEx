from __future__ import annotations

import numpy as np
import pandas as pd
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
from ..utils.logger import logger

class BiasExtension(Extension):
    """Base class for estimating selection bias after matching using linear regression.
    
    This extension quantifies the residual bias between treatment and control groups 
    that remains after the matching procedure. It uses a linear regression model 
    trained on the matched sample to predict the counterfactual outcome, then 
    computes the difference between the observed and predicted values.
    
    The bias is defined as:
        - For treatment group: bias_t = (X - X_matched) * coefficients_t
        - For control group:   bias_c = (X - X_matched) * coefficients_c
        
    Subclasses must implement backend-specific logic for Pandas and Spark.
    
    Attributes:
        grouping_role: Role defining the treatment assignment column.
        target_roles: Role(s) defining the target outcome column(s).
        target_field: Resolved name of the target column.
        group_field: Resolved name of the grouping column.
        features: List of feature column names used for regression.
    """
    def __init__(
            self,
            grouping_role: ABCRole,
            target_roles: list[ABCRole],
    ):
        """Initialize the BiasExtension.
        
        Args:
            grouping_role: The role identifying the treatment/control grouping column.
            target_roles: The role(s) identifying the target outcome column(s).
        """
        super().__init__()
        self.grouping_role = grouping_role
        self.target_roles = target_roles

        self.target_field = None
        self.group_field = None
        self.features = None

    def _set_columns(self, data: Dataset) -> list[str]:
        """Resolve and store target and group field names from the dataset.
        
        Args:
            data: The input dataset to search for roles.
            
        Returns:
            List of resolved column names (target and group).
        """
        self.target_field = data.search_columns(self.target_roles)[0]
        self.group_field = data.search_columns(self.grouping_role)[0]

    @staticmethod
    def prepare_data(
        data: ExperimentData
    ) -> Dataset:
        """Prepare matched data from experiment data (backend-specific)."""
        raise NotImplementedError

    @staticmethod
    def calc_bias(
            X: Dataset, X_matched: Dataset, coefficients: np.ndarray[float]
    ):
        """Calculate bias using feature differences and regression coefficients."""
        raise NotImplementedError
    
    def calc(self, data: Dataset, **kwargs):
        """Execute the full bias estimation pipeline."""
        raise NotImplementedError
    
    def _calc_coefs(self, data: Dataset) -> np.ndarray:
        """Compute linear regression coefficients for each group."""
        raise NotImplementedError
    
    @staticmethod
    def _extract_info(data: Dataset) -> tuple[Dataset, list[str], list[str]]:
        """Extract neighbor indices and numeric columns from the dataset.
        
        Args:
            data: The dataset containing matching results and features.
            
        Returns:
            A tuple containing:
                - List of column names containing neighbor indices.
                - List of numeric column names (features and targets).
                
        Raises:
            ValueError: If no matching index columns are found.
        """
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

@backend_factory.register(BiasExtension, PandasDataset)
class PandasBisaExtesion(BiasExtension):
    """Pandas backend implementation for bias estimation.
    
    Performs in-memory ordinary least squares (OLS) regression using 
    `numpy.linalg.lstsq` to estimate coefficients, and vectorized 
    NumPy operations to compute the final bias adjustments.
    """
    @staticmethod
    def _prepare_data(
        data: Dataset,
        neighbors_cols: list[str] | str,
        numeric_cols: list[str] | str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare matched features for bias estimation using Pandas.
        
        Unstacks the neighbor indices from wide to long format, fetches the 
        corresponding features for each matched neighbor, and aggregates them 
        by computing the mean across all neighbors for each initial observation.
        
        Args:
            data: The input dataset containing features and neighbor indices.
            neighbors_cols: Column name(s) containing the indices of matched neighbors.
            numeric_cols: Numeric column names (features and target) to aggregate.
            
        Returns:
            A tuple containing:
                - The original neighbor indices DataFrame.
                - A DataFrame with aggregated matched features, suffixed with '_matched'.
        """
        neighbors_cols = Adapter.to_list(neighbors_cols)
        numeric_cols = Adapter.to_list(numeric_cols)
        
        t_data = data[numeric_cols].data
        indexes = data[neighbors_cols].data
        
        # Melt the neighbor indexes to long format
        melted = indexes.stack().reset_index()
        melted.columns = ['initial_index', 'neighbor_col', 'match_index']
        melted = melted.dropna(subset=['match_index'])
        
        # Fetch the features of the matched units
        matched_features = t_data.loc[melted['match_index']].copy()
        matched_features.index = melted['initial_index'].values
        
        # Group by original index and calculate mean
        matched_data = matched_features.groupby(level=0).mean()
        matched_data = matched_data.rename(columns={col: f"{col}_matched" for col in numeric_cols})
        
        return indexes, matched_data

    def _calc_coefs(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate linear regression coefficients for each group using OLS.
        
        Fits a separate linear regression model (target_matched ~ features_matched) 
        for the control and treatment groups using `numpy.linalg.lstsq`. 
        An intercept term is included during fitting but discarded in the final 
        output, as the bias calculation relies only on feature differences.
        
        Args:
            data: DataFrame containing original and matched features/targets, 
                  along with the grouping column.
                  
        Returns:
            A numpy array of shape (2, n_features) containing the regression 
            coefficients for group 1 (control) and group 2 (treatment).
        """
        group_1, group_2, *_ = data[self.group_field].unique()
        
        features = [col + "_matched" for col in self.features]
        target = self.target_field + "_matched"
        
        def _get_weights(group_data: pd.DataFrame) -> np.ndarray:
            X = group_data[features].values
            y = group_data[target].values
            
            # Добавляем столбец единиц для интерсепта (аналогично LstsqExtension)
            X_with_intercept = np.c_[np.ones(X.shape[0]), X]
            
            # Решаем задачу МНК (метод наименьших квадратов)
            weights, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
            
            # Отбрасываем первый вес (интерсепт), возвращаем только коэффициенты при фичах
            return weights[1:]

        # Group 1
        fit_data_1 = data[data[self.group_field] == group_1]
        weights_1 = _get_weights(fit_data_1)
        
        # Group 2
        fit_data_2 = data[data[self.group_field] == group_2]
        weights_2 = _get_weights(fit_data_2)
        
        return np.array([weights_1, weights_2])

    def _calc_bias(
        self,
        data: pd.DataFrame,
        coefficients_1: np.ndarray,
        coefficients_2: np.ndarray
    ) -> pd.DataFrame:
        """Compute the final bias adjustment for each observation.
        
        Calculates the dot product between the feature differences 
        (X - X_matched) and the group-specific regression coefficients.
        
        Args:
            data: DataFrame containing original and matched features.
            coefficients_1: Regression coefficients for the control group.
            coefficients_2: Regression coefficients for the treatment group.
            
        Returns:
            A DataFrame indexed by the original observation index, containing 
            the calculated 'bias' and the 'matched_target' values.
        """
        group_1, group_2, *_ = data[self.group_field].unique()
        
        bias = np.zeros(len(data))
        
        mask_1 = data[self.group_field] == group_1
        mask_2 = data[self.group_field] == group_2
        
        features = self.features
        matched_features = [col + "_matched" for col in features]
        
        # Calculate bias for group 1
        if mask_1.any():
            diff_1 = data.loc[mask_1, features].values - data.loc[mask_1, matched_features].values
            bias[mask_1] = np.dot(diff_1, coefficients_1)
            
        # Calculate bias for group 2
        if mask_2.any():
            diff_2 = data.loc[mask_2, features].values - data.loc[mask_2, matched_features].values
            bias[mask_2] = np.dot(diff_2, coefficients_2)
            
        final_data = pd.DataFrame({
            "index": data.index,
            "bias": bias,
            "matched_target": data[self.target_field + "_matched"].values
        })
        final_data.set_index("index", inplace=True)
        
        return final_data

    def calc(self, data: Dataset, **kwargs) -> Dataset:
        """Execute the full bias estimation pipeline for Pandas datasets.
        
        Orchestrates data preparation, coefficient estimation, and bias 
        calculation, returning the results wrapped in a Dataset object.
        
        Args:
            data: The input dataset with matched indices and features.
            **kwargs: Additional arguments (ignored).
            
        Returns:
            A Dataset containing the 'bias' and 'matched_target' columns, 
            indexed by the original observation IDs.
        """
        self._set_columns(data)
        neighbors_cols, numeric_cols = self._extract_info(data)
        
        self.features = [
            col for col in numeric_cols 
            if col != self.group_field and col != self.target_field
        ]
        
        _, matched_data = self._prepare_data(
            data=data,
            neighbors_cols=neighbors_cols,
            numeric_cols=numeric_cols
        )
        
        initial_data = data[numeric_cols + [self.group_field]].data
        initial_data = initial_data.join(matched_data, how='left')
        
        coefficients_1, coefficients_2 = self._calc_coefs(initial_data)
        final_data = self._calc_bias(initial_data, coefficients_1, coefficients_2)
        
        final_dataset = Dataset(
            roles={"bias": InfoRole(), "matched_target": InfoRole()}, 
            data=final_data
        )
        
        return final_dataset

@logger.log_methods(log_args=False, log_result=False, private=True, static=True)
@backend_factory.register(BiasExtension, SparkDataset)
class SparkBisaExtesion(BiasExtension):
    """Spark backend implementation for distributed bias estimation.
    
    Leverages PySpark's distributed DataFrame operations and MLlib's 
    LinearRegression to fit models and compute bias adjustments across 
    large-scale datasets partitioned by the grouping column.
    
    Attributes:
        STORAGE_DICT: dict with storages that are used in HypEx 
        for caching Datasets.
    """

    STORAGE_DICT = {
        "MEMORY_ONLY": StorageLevel.MEMORY_ONLY,
        "MEMORY_AND_DISK": StorageLevel.MEMORY_AND_DISK,
        "DISK_ONLY": StorageLevel.DISK_ONLY,
    }

    @staticmethod
    def _prepare_data(
        data: Dataset, 
        neighbors_cols: list[str] | str, 
        numeric_cols: list[str] | str
    ) -> SparkDF:
        """Prepare matched features for bias estimation using PySpark.
        
        Explodes the neighbor indices, joins with the original feature data, 
        and aggregates by computing the mean of matched features for each 
        initial observation.
        
        Args:
            data: The input dataset.
            neighbors_cols: Column name(s) containing neighbor indices.
            numeric_cols: Numeric columns to aggregate.
            
        Returns:
            A SparkDF with aggregated matched features suffixed with '_matched'.
        """
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
        """Public wrapper for data preparation, returning Dataset objects.
        
        Args:
            data: The input dataset.
            
        Returns:
            A tuple containing the neighbor indices Dataset and the matched data Dataset.
        """
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
        """Distributed linear regression coefficient estimation using MLlib.
        
        Fits a separate Spark LinearRegression model for each group. Data is 
        repartitioned by the group field to optimize distributed training.
        
        Args:
            data: SparkDF containing matched features and targets.
            
        Returns:
            A numpy array of shape (2, n_features) with coefficients for both groups.
        """
        group_1, group_2, *_ = map(
            lambda row: row[0], 
            data.select(self.group_field).distinct().collect()
        )
        features = [col + "_matched" for col in self.features]
        asembler = VectorAssembler(inputCols=features, outputCol='_features')
        lr = LinearRegression(
            featuresCol='_features', 
            labelCol=self.target_field + "_matched", 
            regParam=0.01
        )
        data = data.repartition(F.col(self.group_field))

        fit_data_1 = data.filter(F.col(self.group_field) == group_1)
        fit_data_1 = asembler.transform(fit_data_1)
        fit_data_1.persist()
        model_1 = lr.fit(fit_data_1)
        fit_data_1.unpersist()

        fit_data_2 = data.filter(F.col(self.group_field) == group_2)
        fit_data_2 = asembler.transform(fit_data_2)
        fit_data_2.persist()
        model_2 = lr.fit(fit_data_2)
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
        """Compute bias adjustments using Spark SQL expressions.
        
        Applies conditional logic based on group membership to calculate the 
        dot product of feature differences and regression coefficients.
        
        Args:
            data: SparkDF with original and matched features.
            coefficients_1: Coefficients for the control group.
            coefficients_2: Coefficients for the treatment group.
            
        Returns:
            A SparkDF containing 'index', 'bias', and 'matched_target' columns.
        """
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
        """Execute the full distributed bias estimation pipeline.
        
        Handles caching, model fitting, and bias computation across Spark 
        partitions, ensuring resources are properly released after execution.
        
        Args:
            data: The input dataset with matched indices and features.
            **kwargs: Additional arguments (ignored).
            
        Returns:
            A persisted Dataset containing the 'bias' and 'matched_target' columns.
        """
        storage_level = data.get_storage_level() or "MEMORY_AND_DISK"
        self._set_columns(data)
        neighbors_cols, numeric_cols = self._extract_info(data)
        self.features = [
                col for col in numeric_cols 
                if col != self.group_field and col != self.target_field
        ]

        matched_data = self._prepare_data(
            data = data,
            neighbors_cols=neighbors_cols,
            numeric_cols=numeric_cols
        )
        matched_data.persist(self.STORAGE_DICT[storage_level])
        matched_data.count()

        initial_data: SparkDF = data[numeric_cols + [self.group_field]].data.to_spark(index_col='initial_index')
        initial_data = initial_data.join(matched_data, on='initial_index')
        initial_data.persist(self.STORAGE_DICT[storage_level])
        initial_data.count()

        coefficients_1, coefficients_2  = self._calc_coefs(initial_data)
        final_data = self._calc_bias(initial_data, coefficients_1, coefficients_2)
        final_dataset: Dataset = self.result_to_dataset(final_data, {}, small=False).set_index("index")
        final_dataset.index.name = None

        final_dataset.persist(storage_level)
        initial_data.unpersist()
        matched_data.unpersist()

        return final_dataset