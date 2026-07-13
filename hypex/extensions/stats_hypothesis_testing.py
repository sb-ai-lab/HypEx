from __future__ import annotations
from typing import Any

import numpy as np

from ..dataset import Dataset
from .abstract import Extension
from ..utils import timeit, NAME_BORDER_SYMBOL


class StatsAggregationExtension(Extension):
    """Extension that computes aggregated statistics per group for hypothesis testing.

    Supports both Pandas and Spark backends. The result is a nested dictionary
    structured as ``{group_key: {column: {statistic: value}}}``, which is then
    consumed by stats-based comparators (e.g. :class:`StatsTTest`,
    :class:`StatsChi2Test`) to perform pairwise hypothesis tests without
    re-scanning the raw data.

    Supported statistics: ``mean``, ``std``, ``var``, ``count``, ``sum``,
    ``min``, ``max``.
    """
    
    def _calc_pandas(
        self,
        data: Dataset,
        group_cols: list[str],
        target_cols: list[str],
        stats: list[str],
        **kwargs,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Compute aggregated statistics using the Pandas backend.

        Uses ``DataFrame.groupby().agg()`` with a list of statistics, then
        flattens the resulting ``MultiIndex`` columns into a single level
        using ``NAME_BORDER_SYMBOL`` as the separator.

        Args:
            data: The input dataset to aggregate.
            group_cols: List of column names to group by.
            target_cols: List of target column names to compute statistics for.
            stats: List of statistic names to compute (e.g. ``['mean', 'std']``).
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            A nested dictionary mapping ``group_key`` → ``column`` →
            ``statistic`` → ``value``.
        """
        from ..utils import NAME_BORDER_SYMBOL
        
        pdf = data.data  # pd.DataFrame
        grouped = pdf.groupby(group_cols)
        agg_result = grouped[target_cols].agg(stats)
        
        # Flatten MultiIndex columns: (col, stat) → "col┆stat"
        agg_result.columns = [
            f"{col}{NAME_BORDER_SYMBOL}{stat}"
            for col, stat in agg_result.columns
        ]
        
        result = {}
        for group_key in agg_result.index:
            result[group_key] = {}
            row = agg_result.loc[group_key]
            for col in target_cols:
                result[group_key][col] = {}
                for stat in stats:
                    col_name = f"{col}{NAME_BORDER_SYMBOL}{stat}"
                    result[group_key][col][stat] = row[col_name]
        
        return result
    
    @timeit(level="SPARK", prefix="AGG_EXT_SPARK")
    def _calc_spark(
        self,
        data: Dataset,
        group_cols: list[str],
        target_cols: list[str],
        stats: list[str],
        **kwargs,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Compute aggregated statistics using the Spark backend.

        Builds explicit PySpark aggregation expressions for each
        ``(column, statistic)`` pair and executes a single ``groupBy().agg()``
        job. Each expression is aliased as ``column┆stat`` for easy retrieval.

        Args:
            data: The input dataset to aggregate.
            group_cols: List of column names to group by.
            target_cols: List of target column names to compute statistics for.
            stats: List of statistic names to compute (e.g. ``['mean', 'std']``).
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            A nested dictionary mapping ``group_key`` → ``column`` →
            ``statistic`` → ``value``. For single-column grouping, the
            ``group_key`` is a scalar; for multi-column grouping, it is a
            tuple of values.
        """
        import pyspark.sql.functions as F
        
        sdf = data.data.to_spark()
        
        # Строим все agg expressions
        agg_exprs = []
        for col in target_cols:
            for stat in stats:
                alias = f"{col}{NAME_BORDER_SYMBOL}{stat}"
                if stat == "mean":
                    agg_exprs.append(F.mean(F.col(col)).alias(alias))
                elif stat == "std":
                    agg_exprs.append(F.stddev(F.col(col)).alias(alias))
                elif stat == "var":
                    agg_exprs.append(F.variance(F.col(col)).alias(alias))
                elif stat == "count":
                    agg_exprs.append(F.count(F.col(col)).alias(alias))
                elif stat == "sum":
                    agg_exprs.append(F.sum(F.col(col)).alias(alias))
                elif stat == "min":
                    agg_exprs.append(F.min(F.col(col)).alias(alias))
                elif stat == "max":
                    agg_exprs.append(F.max(F.col(col)).alias(alias))
        
        agg_sdf = sdf.groupBy(*group_cols).agg(*agg_exprs)
        rows = agg_sdf.collect()
        
        result = {}
        for row in rows:
            group_key = (
                row[group_cols[0]] 
                if len(group_cols) == 1 
                else tuple(row[c] for c in group_cols)
            )
            result[group_key] = {}
            for col in target_cols:
                result[group_key][col] = {}
                for stat in stats:
                    alias = f"{col}{NAME_BORDER_SYMBOL}{stat}"
                    result[group_key][col][stat] = row[alias]
        
        return result


class StatsKSTestExtension(Extension):
    """Extension that prepares per-group histograms for the Kolmogorov-Smirnov test.

    For the Pandas backend, returns empty histograms with group counts — the
    actual KS computation is delegated to :class:`GroupKSTestExtension`
    (``scipy.stats.ks_2samp``), which is faster for small in-memory data.

    For the Spark backend, computes global min/max bounds, unpivots all
    target columns via ``F.explode``, and builds per-group histograms using
    a fixed number of bins (``n_bins``). This requires only two Spark jobs
    regardless of the number of target columns.
    """
    
    def __init__(self, n_bins: int = 2000, reliability: float = 0.05):
        """Initializes the KS test extension.

        Args:
            n_bins: Number of histogram bins to use for discretizing the
                continuous distribution in the Spark path. Defaults to 2000.
            reliability: Significance level (alpha) for the KS test.
                Defaults to 0.05.
        """
        super().__init__()
        self.n_bins = n_bins
        self.reliability = reliability
    
    def _calc_pandas(
        self,
        data: Dataset,
        group_col: str,
        target_cols: list[str],
        **kwargs,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Prepare per-group histograms using the Pandas backend.

        Returns empty histograms with group counts — the actual KS test
        is performed later by :class:`GroupKSTestExtension` on raw data,
        which is more efficient for in-memory Pandas datasets.

        Args:
            data: The input dataset.
            group_col: Name of the column to group by.
            target_cols: List of target column names.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            A nested dictionary mapping ``group_key`` → ``column`` →
            ``{"histogram": {}, "count": n}``.
        """
        from ..extensions.scipy_stats import GroupKSTestExtension
        
        ext = GroupKSTestExtension(reliability=self.reliability, n_bins=self.n_bins)
        
        result = {}
        grouped = data.data.groupby(group_col)
        
        for group_key, group_df in grouped:
            result[group_key] = {}
            group_ds = Dataset(
                roles={col: data.roles.get(col) for col in target_cols},
                data=group_df[target_cols],
            )
            for col in target_cols:
                col_ds = group_ds[[col]]
                result[group_key][col] = {
                    "histogram": {},
                    "count": len(col_ds),
                }
        
        return result
    
    def _calc_spark(
        self,
        data: Dataset,
        group_col: str,
        target_cols: list[str],
        **kwargs,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Compute per-group histograms using the Spark backend.

        Executes two Spark jobs:

        1. **Global bounds**: computes ``min`` and ``max`` for all target
           columns in a single ``agg()`` call.
        2. **Histograms**: unpivots all target columns via ``F.explode``,
           assigns each value to a bin using ``floor((value - min) / width)``,
           and counts per ``(group, bucket)`` pair.

        Edge cases (all values identical, or all ``NULL``) are handled by
        returning a single-bin histogram with the group count.

        Args:
            data: The input dataset.
            group_col: Name of the column to group by.
            target_cols: List of target column names.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            A nested dictionary mapping ``group_key`` → ``column`` →
            ``{"histogram": {bucket: count, ...}, "count": n}``.
        """
        
        import pyspark.sql.functions as F
        
        sdf = data.data.to_spark()
        
        agg_exprs = []
        for col in target_cols:
            agg_exprs.extend([
                F.min(F.col(col)).alias(f"{col}┆min"),
                F.max(F.col(col)).alias(f"{col}┆max"),
            ])
        bounds_row = sdf.agg(*agg_exprs).collect()[0]
        
        unpivoted = sdf.select(
            group_col,
            F.explode(F.array([
                F.struct(
                    F.lit(col).alias("column_name"),
                    F.col(col).alias("value")
                ) for col in target_cols
            ])).alias("data")
        ).select(
            group_col,
            F.col("data.column_name").alias("column_name"),
            F.col("data.value").alias("value")
        )
        
        all_group_stats = {}
        
        for col in target_cols:
            col_min = bounds_row[f"{col}┆min"]
            col_max = bounds_row[f"{col}┆max"]
            
            # Counts per group
            count_df = unpivoted.filter(
                (F.col("column_name") == col) & F.col("value").isNotNull()
            ).groupBy(group_col).count().collect()
            group_counts = {row[group_col]: row["count"] for row in count_df}
            
            if col_min is None or col_max is None or col_min == col_max:
                for grp, cnt in group_counts.items():
                    all_group_stats.setdefault(grp, {})[col] = {
                        "histogram": {0: cnt} if cnt > 0 else {},
                        "count": cnt,
                    }
                continue
            
            width = (col_max - col_min) / self.n_bins
            
            col_data = unpivoted.filter(F.col("column_name") == col)
            unpivoted_with_bucket = col_data.withColumn(
                "_bucket",
                F.least(
                    F.floor((F.col("value") - F.lit(col_min)) / F.lit(width)),
                    F.lit(self.n_bins - 1),
                ).cast("int"),
            )
            
            hist_rows = (
                unpivoted_with_bucket
                .filter(F.col("value").isNotNull())
                .groupBy(group_col, "_bucket")
                .count()
                .collect()
            )
            
            for row in hist_rows:
                grp = row[group_col]
                bucket = int(row["_bucket"])
                count = row["count"]
                all_group_stats.setdefault(grp, {}).setdefault(col, {
                    "histogram": {},
                    "count": group_counts.get(grp, 0),
                })
                all_group_stats[grp][col]["histogram"][bucket] = count
            
            for grp, cnt in group_counts.items():
                if grp not in all_group_stats:
                    all_group_stats[grp] = {}
                if col not in all_group_stats[grp]:
                    all_group_stats[grp][col] = {"histogram": {}, "count": cnt}
        
        return all_group_stats


class StatsChi2TestExtension(Extension):
    """Extension that prepares per-group value counts for the chi-squared test.

    For the Pandas backend, uses ``groupby().value_counts()`` to compute
    category frequencies per group.

    For the Spark backend, unpivots all target columns via ``F.explode``
    and computes value counts in a single ``groupBy().count()`` job,
    regardless of the number of target columns.
    """
    
    def __init__(self, reliability: float = 0.05):
        """Initializes the chi-squared test extension.

        Args:
            reliability: Significance level (alpha) for the chi-squared test.
                Defaults to 0.05.
        """
        super().__init__()
        self.reliability = reliability
    
    def _calc_pandas(
        self,
        data: Dataset,
        group_col: str,
        target_cols: list[str],
        **kwargs,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Compute per-group value counts using the Pandas backend.

        Iterates over groups produced by ``DataFrame.groupby(group_col)``
        and computes ``value_counts()`` for each target column.

        Args:
            data: The input dataset.
            group_col: Name of the column to group by.
            target_cols: List of target column names.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            A nested dictionary mapping ``group_key`` → ``column`` →
            ``{"value_counts": {category: count, ...}}``.
        """
        result = {}
        grouped = data.data.groupby(group_col)
        
        for group_key, group_df in grouped:
            result[group_key] = {}
            for col in target_cols:
                vc = group_df[col].value_counts().to_dict()
                result[group_key][col] = {"value_counts": vc}
        
        return result
    
    def _calc_spark(
        self,
        data: Dataset,
        group_col: str,
        target_cols: list[str],
        **kwargs,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Compute per-group value counts using the Spark backend.

        Unpivots all target columns via ``F.explode(F.array(...))`` and
        computes value counts in a single ``groupBy(group_col, column_name, value).count()``
        job. ``NULL`` values are filtered out before aggregation.

        Args:
            data: The input dataset.
            group_col: Name of the column to group by.
            target_cols: List of target column names.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            A nested dictionary mapping ``group_key`` → ``column`` →
            ``{"value_counts": {category: count, ...}}``.
        """
        import pyspark.sql.functions as F
        
        sdf = data.data.to_spark()
        
        # UNPIVOT
        unpivoted = sdf.select(
            group_col,
            F.explode(F.array([
                F.struct(
                    F.lit(col).alias("column_name"),
                    F.col(col).alias("value")
                ) for col in target_cols
            ])).alias("data")
        ).select(
            group_col,
            F.col("data.column_name").alias("column_name"),
            F.col("data.value").alias("value")
        )
        
        value_counts_df = (
            unpivoted
            .filter(F.col("value").isNotNull())
            .groupBy(group_col, "column_name", "value")
            .count()
            .collect()
        )
        
        result = {}
        for row in value_counts_df:
            grp = row[group_col]
            col = row["column_name"]
            val = row["value"]
            cnt = row["count"]
            result.setdefault(grp, {}).setdefault(col, {"value_counts": {}})
            result[grp][col]["value_counts"][val] = cnt
        
        return result