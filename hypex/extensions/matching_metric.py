from __future__ import annotations

import numpy as np
import pandas as pd

from pyspark import StorageLevel
from pyspark.sql import (
    functions as F,
    DataFrame as SparkDF,
)
from typing import Literal

from .abstract import Extension
from .scipy_stats import NormCDF

from ..utils.registry import backend_factory
from ..utils import Adapter
from ..dataset.backends import PandasDataset, SparkDataset
from ..dataset import (
    Dataset, 
    SmallDataset,
    ABCRole, 
    FeatureRole, 
    TargetRole, 
    AdditionalMatchingRole, 
    AdditionalStatisticRole,
    AdditionalTargetRole,
    InfoRole
)

# TODO: logger
from ..utils.logger import logger


class MatchingMetricsExtension(Extension):
    PERSIST_POLITIC = StorageLevel.MEMORY_AND_DISK
    def __init__(
            self,
            grouping_role: ABCRole,
            target_roles: ABCRole | list[ABCRole],
            metric: Literal["auto", "atc", "att", "ate"],
            n_neighbors: int,
        ):
        super().__init__()
        self.grouping_role = grouping_role
        self.target_roles = target_roles
        self.metric = metric
        self.n_neighbors = n_neighbors
        
        self.new_target_field = None
        self.neighbors_cols = None

    
    def _calc_stats_and_weights(self, data):
        raise NotADirectoryError
    
    def prepare_data(self, data: Dataset) -> Dataset:
        neighbors_cols, _, _, _ = self._extract_info(data)
        matched_data = self._prepare_data(
            data = data,
            neighbors_cols=neighbors_cols,
            numeric_cols=self.target_field
        )
        matched_data = self.result_to_dataset(matched_data, {}, small=False)
        matched_data = matched_data.set_index('initial_index')
        matched_data.index.name = None

        return matched_data
    
    @staticmethod
    def _prepare_data(
        data: Dataset, 
        neighbors_cols: list[str] | str, 
        numeric_cols: list[str] | str
    ):
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
        bias_col = data.search_columns(AdditionalStatisticRole())[0] or None
        new_target_col = data.search_columns(AdditionalTargetRole())[0] or None
        return neighbors_cols, numeric_cols, bias_col, new_target_col
    
    def _set_columns(self, data: Dataset) -> list[str]:
        self.target_field = data.search_columns(self.target_roles)[0]
        self.group_field = data.search_columns(self.grouping_role)[0]
    
    @staticmethod
    def _calc_se(
        n_c: int, n_t: int, var_c: float, var_t: float, w_c: float, w_t: float
    ) -> float:
        return np.sqrt(w_c * var_c / n_c ** 2 + w_t * var_t / n_t ** 2)
    
    @staticmethod
    def _calc_p_value(
            x: float
    ) -> float:
        return (
            NormCDF()
            .calc(
                SmallDataset.from_dict(
                    {"value": [x]}, roles={"value": InfoRole()}
                )
            )
            .get_values()[0][0]
        )
    
    def _calc_metrics(
            self, 
            stats_itc: dict[str, float], 
            stats_itt: dict[str, float]
    ) -> dict[str, float]:
        itt_se = self._calc_se(
            n_c=stats_itc['count'], n_t=stats_itt['count'], 
            var_c=stats_itc['var'], var_t=stats_itt['var'], 
            w_c=stats_itt['count'],
            w_t=(stats_itt['count'] / stats_itc['count']) ** 2 * stats_itt["sq_sum"]
        )
        itc_se = self._calc_se(
            n_c=stats_itt['count'], n_t=stats_itc['count'], 
            var_c=stats_itt['var'], var_t=stats_itc['var'],
            w_c=(stats_itc['count'] / (stats_itt['count'])) ** 2 * stats_itc["sq_sum"],
            w_t=stats_itc['count']
        )

        p_val_itt = self._calc_p_value(stats_itt['mean'] / itt_se)
        p_val_itc = self._calc_p_value(stats_itc['mean'] / itc_se)

        if self.metric == "atc":
            return {
                "ATC": [
                    stats_itc['mean'] , itc_se, p_val_itc,
                    stats_itc['mean']  - 1.96 * itc_se, 
                    stats_itc['mean']  + 1.96 * itc_se,
                ]
            }
        if self.metric == "att":
            return {
                "ATC": [
                    stats_itt['mean'] , itt_se, p_val_itt,
                    stats_itt['mean']  - 1.96 * itt_se, 
                    stats_itt['mean']  + 1.96 * itt_se,
                ]
            }
        ate = (
            (
                stats_itt['mean']  * stats_itt['count'] + 
                stats_itc['mean'] * stats_itc['count']
            ) / (stats_itt['count'] + stats_itc['count'])
        )
        ate_se = self._calc_se(
            n_c=stats_itc['count'] + stats_itt['count'], n_t=stats_itc['count'] + stats_itt['count'], 
            var_c=stats_itc['var'], var_t=stats_itt['var'],
            w_c=stats_itc['count'] + 2 * stats_itc["sum"] + stats_itc["sq_sum"],
            w_t=stats_itt['count'] + 2 * stats_itt["sum"] + stats_itt["sq_sum"]
        )
        p_val_ate = self._calc_p_value(ate / ate_se)
        return {
            "ATT": [
                    stats_itt['mean'] , itt_se, p_val_itt,
                    stats_itt['mean']  - 1.96 * itt_se, 
                    stats_itt['mean']  + 1.96 * itt_se,
                ],
            "ATC": [
                    stats_itc['mean'] , itc_se, p_val_itc,
                    stats_itc['mean']  - 1.96 * itc_se, 
                    stats_itc['mean']  + 1.96 * itc_se,
                ],
            "ATE": [
                    ate, ate_se, p_val_ate, 
                    ate - 1.96 * ate_se, 
                    ate + 1.96 * ate_se
                ],
        }
    
    def calc(self, data: Dataset, **kwargs):
        self._set_columns(data)
        neighbors_cols, numeric_cols, bias_col, new_target_col = self._extract_info(data)
        if bias_col is None:
            # neighbors_cols, numeric_cols, bias_col = self._extract_info(data)
            new_target_data = self.prepare_data(
                data = data,
                neighbors_cols=neighbors_cols,
                numeric_cols=numeric_cols
            )
            new_data = data.add_column(new_target_data)
            new_target_col = self.target_field + "_matched"
            bias_col = "bias"
        else:
            new_data = data

        self.neighbors_cols = neighbors_cols
        self.new_target_field = new_target_col
        self.bias_field = bias_col

        stats_itc, stats_itt = self._calc_stats_and_weights(new_data)
        return self._calc_metrics(stats_itc, stats_itt)
  

@backend_factory.register(MatchingMetricsExtension, PandasDataset)
class PandasMatchingMetricsExtension(MatchingMetricsExtension):
    
    @staticmethod
    def _prepare_data(
        data: Dataset, 
        neighbors_cols: list[str] | str, 
        numeric_cols: list[str] | str
    ) -> pd.DataFrame:
        neighbors_cols = Adapter.to_list(neighbors_cols)
        numeric_cols = Adapter.to_list(numeric_cols)
        
        t_data = data[numeric_cols].data
        indexes = data[neighbors_cols].data
        
        # "expand" the neighbor indexes from a wide format to a long one
        melted = indexes.stack().reset_index()
        melted.columns = ['initial_index', 'neighbor_col', 'match_index']
        
        # adjusting the features of our neighbors according to their indexes
        matched_features = t_data.loc[melted['match_index']].copy()
        matched_features.index = melted['initial_index'].values
        
        # calc mean by initial index
        matched_data = matched_features.groupby(level=0).mean()
        matched_data = matched_data.rename(columns={col: f"{col}_matched" for col in numeric_cols})
        
        # add zero bias if Bias extension didn't execute
        matched_data['bias'] = 0.0
        
        return matched_data

    @staticmethod
    def _calc_scaled_counts(
        data: pd.DataFrame,
        match_idx_cols: str | list[str],
        n_neighbors: int
    ) -> pd.Series:
        match_idx_cols = Adapter.to_list(match_idx_cols)
        
        all_neighbors = pd.Series(
            data[match_idx_cols].values.flatten()
        )
        
        scaled_counts = all_neighbors.value_counts() / n_neighbors
        scaled_counts.name = "scaled_counts"
        return scaled_counts

    def _calc_stats_and_weights(self, data: Dataset) -> tuple[dict[str, float], dict[str, float]]: 
        new_data: pd.DataFrame = data.data.copy()
        scaled_counts = self._calc_scaled_counts(new_data, self.neighbors_cols, self.n_neighbors)
        
        group_1, group_2, *_ = new_data[self.group_field].unique()
        
        # Individual Treatment effect (_it) vectorized calc using numpy!
        _it = np.zeros(len(new_data))
        
        mask_1 = new_data[self.group_field] == group_1
        mask_2 = new_data[self.group_field] == group_2
        
        target_vals = new_data[self.target_field].values
        new_target_vals = new_data[self.new_target_field].values
        bias_vals = new_data[self.bias_field].values
        
        # control (group_1): target - matched_target - bias
        _it[mask_1] = target_vals[mask_1] - new_target_vals[mask_1] - bias_vals[mask_1]
        # test (group_2): target - matched_target + bias
        _it[mask_2] = target_vals[mask_2] - new_target_vals[mask_2] + bias_vals[mask_2]
        
        new_data['_it'] = _it
        
        new_data = new_data.join(scaled_counts, how='left')
        new_data['scaled_counts'] = new_data['scaled_counts'].fillna(0)
        

        stats = (
            new_data
            .groupby(self.group_field)
            .agg(
                count=('_it', 'count'),
                mean=('_it', 'mean'),
                var=('_it', 'var'),
                sum=('scaled_counts', 'sum'),
                sq_sum=('scaled_counts', lambda x: (x ** 2).sum())
            )
            .reset_index()
        )

        stats_dict_1 = stats[stats[self.group_field] == group_1].iloc[0].to_dict()
        stats_dict_1.pop(self.group_field, None) 
        
        stats_dict_2 = stats[stats[self.group_field] == group_2].iloc[0].to_dict()
        stats_dict_2.pop(self.group_field, None)
        
        return stats_dict_1, stats_dict_2
        
@logger.log_methods(log_args=False, log_result=False, private=True, static=True)
@backend_factory.register(MatchingMetricsExtension, SparkDataset)
class SparkMatchingMetricsExtension(MatchingMetricsExtension):
    """
    """

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
                F.explode(F.array(*working_columns)).alias('index')
            )
            .join(other=t_data, on='index')
            .groupBy('initial_index')
            .agg(
                *[
                    F.mean(col).alias(col + "_matched") for col in t_data.columns if col != 'index'
                ]
            )
            .withColumn('bias', F.lit(0))
        )

        return matched_data
    
    @staticmethod
    def _calc_scaled_counts( 
            data: SparkDF, 
            match_idx_cols: str | list[str], 
            n_neighbors: int
    ) -> SparkDF:
        match_idx_cols = Adapter.to_list(match_idx_cols)
        return (
            data
            .select(
                F.explode(F.array(*match_idx_cols)).alias('index')
            )
            .groupBy('index')
            .agg((F.count('index') / n_neighbors).alias('scaled_counts'))
            # .withColumnRenamed('count', 'scaled_counts')
        ) 

    def _calc_stats_and_weights(self, data: Dataset) -> tuple[dict[str, float]]:
        new_data: SparkDF = data.data.to_spark(index_col='index')
        scaled_counts = self._calc_scaled_counts(new_data, self.neighbors_cols, self.n_neighbors) 
        scaled_counts.persist(self.PERSIST_POLITIC)
        # First group is `control`, second one is `test`
        group_1, group_2, *_ = map(
            lambda row: row[0], 
            new_data.select(self.group_field).distinct().collect()
        )  
        stats = (
            new_data
            .select(
                'index',
                self.group_field,
                self.target_field,
                self.new_target_field,
                self.bias_field
            )
            .withColumn(
                '_it', 
                F.when(
                    F.col(self.group_field) == group_1,
                    F.col(self.target_field) - F.col(self.new_target_field) - F.col(self.bias_field)
                )
                .when(
                    F.col(self.group_field) == group_2,
                    F.col(self.target_field) - F.col(self.new_target_field) + F.col(self.bias_field)
                )
                .otherwise(0)
            )
            .join(scaled_counts, on='index', how='left')
            .fillna(0)
            .groupBy(self.group_field)
            .agg(
                F.count('_it').alias('count'),
                F.mean('_it').alias('mean'),
                (F.std('_it') ** 2).alias('var'),
                F.sum('scaled_counts').alias('sum'),
                (F.sum(F.col('scaled_counts') ** 2)).alias('sq_sum')
            )
            .toPandas()
        )

        stats_dict_1 = stats[stats[self.group_field] == group_1].iloc[0].to_dict()
        # Del group column
        stats_dict_1.pop(self.group_field, None) 

        stats_dict_2 = stats[stats[self.group_field] == group_2].iloc[0].to_dict()
        stats_dict_2.pop(self.group_field, None)
        
        scaled_counts.unpersist()
        return stats_dict_1, stats_dict_2         