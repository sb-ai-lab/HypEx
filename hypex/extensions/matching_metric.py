from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import (
    DataFrame as SparkDF,
)
from pyspark.sql import (
    functions as F,
)

from ..dataset import (
    ABCRole,
    AdditionalMatchingRole,
    AdditionalStatisticRole,
    AdditionalTargetRole,
    Dataset,
    FeatureRole,
    InfoRole,
    SmallDataset,
    TargetRole,
)
from ..dataset.backends import PandasDataset, SparkDataset
from ..utils import Adapter

# TODO: logger
from ..utils.logger import logger
from ..utils.registry import backend_factory
from .abstract import Extension
from .scipy_stats import NormCDF


class MatchingMetricsExtension(Extension):
    """Base class for estimating treatment effects (ATT, ATC, ATE) and their 
    statistical significance after matching.

    Computes the Individual Treatment Effect on the Treated (ITT) and Control (ITC), 
    applies optional bias correction, and calculates the final Average Treatment 
    Effects along with their standard errors, p-values, and confidence intervals.

    Attributes:
        PERSIST_POLITIC: Default Spark storage level for intermediate data.
        grouping_role: Role identifying the treatment assignment column.
        target_roles: Role(s) identifying the target outcome column(s).
        metric: The type of treatment effect to estimate ('atc', 'att', 'ate', or 'auto').
        n_neighbors: Number of neighbors used in the matching process (for weight scaling).
    """
    PERSIST_POLITIC = StorageLevel.MEMORY_AND_DISK
    def __init__(
            self,
            grouping_role: ABCRole,
            target_roles: ABCRole | list[ABCRole],
            metric: Literal["auto", "atc", "att", "ate"],
            n_neighbors: int,
        ):
        """Initialize the MatchingMetricsExtension.

        Args:
            grouping_role: Role defining the treatment/control grouping column.
            target_roles: Role(s) defining the target outcome column(s).
            metric: Treatment effect type ('atc', 'att', 'ate', or 'auto' for all).
            n_neighbors: Number of matched neighbors per observation.
        """
        super().__init__()
        self.grouping_role = grouping_role
        self.target_roles = target_roles
        self.metric = metric
        self.n_neighbors = n_neighbors

        self.new_target_field = None
        self.neighbors_cols = None


    def _calc_stats_and_weights(self, data):
        """Compute group statistics and neighbor weights (backend-specific)."""
        raise NotADirectoryError

    def prepare_data(self, data: Dataset) -> Dataset:
        """Generate matched target values if bias estimation was skipped.

        Args:
            data: The input dataset.

        Returns:
            A Dataset containing the aggregated matched target values.
        """
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
        """Aggregate matched features/targets (backend-specific)."""
        raise NotImplementedError

    @staticmethod
    def _extract_info(data: Dataset) -> tuple[Dataset, list[str], list[str]]:
        """Extract neighbor indices, numeric columns, bias, and new target columns.

        Args:
            data: The dataset containing matching results.

        Returns:
            A tuple containing:
                - List of neighbor index columns.
                - List of numeric columns.
                - Name of the bias column (or None if not present).
                - Name of the new (matched) target column (or None if not present).
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
        bias_col = data.search_columns(AdditionalStatisticRole())[0] or None
        new_target_col = data.search_columns(AdditionalTargetRole())[0] or None
        return neighbors_cols, numeric_cols, bias_col, new_target_col

    def _set_columns(self, data: Dataset) -> list[str]:
        """Resolve and store target and group field names."""
        self.target_field = data.search_columns(self.target_roles)[0]
        self.group_field = data.search_columns(self.grouping_role)[0]

    @staticmethod
    def _calc_se(
        n_c: int, n_t: int, var_c: float, var_t: float, fs_c: float, fs_t: float
    ) -> float:
        """Calculate the standard error of the treatment effect estimate.

        Args:
            n_c: Sample size of the control group.
            n_t: Sample size of the treatment group.
            var_c: Variance of the individual treatment effect in control.
            var_t: Variance of the individual treatment effect in treatment.
            fs_c: Sum of squared weights for the control group times size of control group.
            fs_t: Sum of squared weights for the treatment group times size of treatment group.

        Returns:
            The computed standard error.
        """
        return np.sqrt(fs_c * var_c / n_c + fs_t * var_t / n_t)

    @staticmethod
    def _calc_p_value(
            x: float
    ) -> float:
        """Calculate a two-sided p-value from a z-score using the normal CDF.

        Args:
            x: The z-score (estimate / standard error).

        Returns:
            The two-sided p-value.
        """
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
        """Compute final ATT, ATC, and ATE metrics with confidence intervals.

        Args:
            stats_itc: Aggregated statistics for the Individual Treatment effect on Control.
            stats_itt: Aggregated statistics for the Individual Treatment effect on Treated.

        Returns:
            A dictionary mapping metric names ('ATT', 'ATC', 'ATE') to lists 
            containing [Estimate, Standard Error, P-value, CI Lower, CI Upper].
        """
        m = stats_itc["count"]
        n = stats_itt["count"]

        var_c = stats_itc["var"]
        var_t = stats_itt["var"]

        sq_c = stats_itc["sq_sum"]
        sq_t = stats_itt["sq_sum"]

        att_se = self._calc_se(
            n_c=m,
            n_t=n,
            var_c=var_c,
            var_t=var_t,
            fs_c=m * sq_c / (n ** 2),
            fs_t=1.0,
        )

        atc_se = self._calc_se(
            n_c=m,
            n_t=n,
            var_c=var_c,
            var_t=var_t,
            fs_c=1.0,
            fs_t=n * sq_t / (m ** 2),
        )

        p_val_att = self._calc_p_value(stats_itt['mean'] / att_se)
        p_val_atc = self._calc_p_value(stats_itc['mean'] / atc_se)

        if self.metric == "atc":
            return {
                "ATC": [
                    stats_itc['mean'] , atc_se, p_val_atc,
                    stats_itc['mean']  - 1.96 * atc_se,
                    stats_itc['mean']  + 1.96 * atc_se,
                ]
            }
        if self.metric == "att":
            return {
                "ATT": [
                    stats_itt['mean'] , att_se, p_val_att,
                    stats_itt['mean']  - 1.96 * att_se,
                    stats_itt['mean']  + 1.96 * att_se,
                ]
            }
        ate = (
            (
                stats_itt['mean']  * stats_itt['count'] +
                stats_itc['mean'] * stats_itc['count']
            ) / (stats_itt['count'] + stats_itc['count'])
        )
        N = m + n

        att_var = att_se ** 2
        atc_var = atc_se ** 2

        ate_var = (n / N) ** 2 * att_var + (m / N) ** 2 * atc_var
        ate_se = float(np.sqrt(max(ate_var, 0.0)))

        # sum_c = float(stats_itc.get("sum", 0.0))
        # sum_t = float(stats_itt.get("sum", 0.0))

        # sq_c = float(stats_itc["sq_sum"])
        # sq_t = float(stats_itt["sq_sum"])

        # ate_var = (
        #     var_c * (m + 2.0 * sum_c + sq_c)
        #     + var_t * (n + 2.0 * sum_t + sq_t)
        # ) / (N ** 2)

        # ate_se = float(np.sqrt(max(ate_var, 0.0)))

        p_val_ate = self._calc_p_value(ate / ate_se)
        return {
            "ATT": [
                    stats_itt['mean'] , att_se, p_val_att,
                    stats_itt['mean']  - 1.96 * att_se,
                    stats_itt['mean']  + 1.96 * att_se,
                ],
            "ATC": [
                    stats_itc['mean'] , atc_se, p_val_atc,
                    stats_itc['mean']  - 1.96 * atc_se,
                    stats_itc['mean']  + 1.96 * atc_se,
                ],
            "ATE": [
                    ate, ate_se, p_val_ate,
                    ate - 1.96 * ate_se,
                    ate + 1.96 * ate_se
                ],
        }

    def calc(self, data: Dataset, **kwargs):
        """Execute the full matching metrics pipeline.

        Resolves columns, prepares matched targets if bias correction is missing, 
        calculates weights and statistics, and returns the final treatment effects.

        Args:
            data: The input dataset containing matched indices, features, and targets.
            **kwargs: Additional arguments (ignored).

        Returns:
            A dictionary of computed treatment effects (ATT, ATC, ATE).
        """
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
    """Pandas backend implementation for matching metrics calculation.

    Uses vectorized NumPy operations and Pandas aggregations to efficiently 
    compute individual treatment effects, neighbor weights, and group statistics 
    in local memory.
    """
    @staticmethod
    def _prepare_data(
        data: Dataset,
        neighbors_cols: list[str] | str,
        numeric_cols: list[str] | str
    ) -> pd.DataFrame:
        """Aggregate matched targets using Pandas.

        Args:
            data: The input dataset.
            neighbors_cols: Columns containing neighbor indices.
            numeric_cols: Numeric columns to aggregate.

        Returns:
            A DataFrame with aggregated matched targets and a default zero bias column.
        """
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
        """Calculate scaled frequency counts of matched neighbors.

        Flattens the neighbor index columns, counts occurrences, and scales 
        by the number of neighbors to derive observation weights.

        Args:
            data: DataFrame containing neighbor index columns.
            match_idx_cols: List of columns containing neighbor indices.
            n_neighbors: The number of neighbors per observation.

        Returns:
            A Pandas Series where the index is the neighbor ID and the value 
            is the scaled weight.
        """
        match_idx_cols = Adapter.to_list(match_idx_cols)

        all_neighbors = pd.Series(
            data[match_idx_cols].values.flatten()
        )

        scaled_counts = all_neighbors.value_counts() / n_neighbors
        scaled_counts.name = "scaled_counts"
        return scaled_counts

    def _calc_stats_and_weights(self, data: Dataset) -> tuple[dict[str, float], dict[str, float]]:
        """Compute individual treatment effects and group statistics using Pandas.

        Calculates the Individual Treatment effect (_it) vectorized via NumPy masks, 
        joins with neighbor weights, and aggregates statistics (mean, variance, sum) 
        per group.

        Args:
            data: The dataset containing targets, matched targets, bias, and groups.

        Returns:
            A tuple of two dictionaries containing statistics for group 1 (control) 
            and group 2 (treatment).
        """
        new_data: pd.DataFrame = data.data.copy()
        scaled_counts = self._calc_scaled_counts(new_data, self.neighbors_cols, self.n_neighbors)

        group_1, group_2, *_ = sorted(new_data[self.group_field].unique())

        # Individual Treatment effect (_it) vectorized calc using numpy!
        _it = np.zeros(len(new_data))

        mask_1 = new_data[self.group_field] == group_1
        mask_2 = new_data[self.group_field] == group_2

        target_vals = new_data[self.target_field].values
        new_target_vals = new_data[self.new_target_field].values
        bias_vals = new_data[self.bias_field].values

        # control (group_1): matched_target -target - bias
        _it[mask_1] = new_target_vals[mask_1] - target_vals[mask_1] - bias_vals[mask_1]
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
    Spark backend implementation for distributed matching metrics calculation.

    Uses Spark SQL functions and distributed aggregations to compute treatment 
    effects and weights across large datasets partitioned by the grouping column.
    """
    @staticmethod
    def _prepare_data(
        data: Dataset,
        neighbors_cols: list[str] | str,
        numeric_cols: list[str] | str
    ) -> SparkDF:
        """Aggregate matched targets using PySpark.

        Args:
            data: The input dataset.
            neighbors_cols: Columns containing neighbor indices.
            numeric_cols: Numeric columns to aggregate.

        Returns:
            A SparkDF with aggregated matched targets and a default zero bias column.
        """
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
        """Calculate scaled frequency counts of matched neighbors using Spark.

        Args:
            data: SparkDF containing neighbor index columns.
            match_idx_cols: List of columns containing neighbor indices.
            n_neighbors: The number of neighbors per observation.

        Returns:
            A SparkDF with 'index' (neighbor ID) and 'scaled_counts' columns.
        """
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
        """Compute individual treatment effects and group statistics using PySpark.

        Calculates the Individual Treatment effect (_it) using Spark SQL conditional 
        expressions, joins with neighbor weights, and aggregates statistics per group.

        Args:
            data: The dataset containing targets, matched targets, bias, and groups.

        Returns:
            A tuple of two dictionaries containing statistics for group 1 (control) 
            and group 2 (treatment).
        """
        new_data: SparkDF = data.data.to_spark(index_col='index')
        scaled_counts = self._calc_scaled_counts(new_data, self.neighbors_cols, self.n_neighbors)
        scaled_counts.persist(self.PERSIST_POLITIC)
        # First group is `control`, second one is `test`
        group_1, group_2, *_ = sorted(
            map(
                lambda row: row[0],
                new_data.select(self.group_field).distinct().collect()
            )
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
                    F.col(self.new_target_field) - F.col(self.target_field) - F.col(self.bias_field)
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
