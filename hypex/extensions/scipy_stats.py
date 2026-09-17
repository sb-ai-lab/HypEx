from __future__ import annotations

import warnings
from typing import Callable, Sequence, Any

import numpy as np
import pandas as pd

from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.sql import DataFrame as SparkDF

from scipy.stats import (  # type: ignore
    chi2_contingency,
    ks_2samp,
    mannwhitneyu,
    norm,
    ttest_ind,
    kstwo,
    kstwobign
)
from ..utils.registry import backend_factory

from ..dataset import SmallDataset, Dataset, DatasetAdapter, StatisticRole
from ..dataset.backends import PandasDataset, SparkDataset
from .abstract import CompareExtension

class GroupStatTest(CompareExtension):
    """
    Master-abstract class for statistic test calculation.
    """
    def __init__(
        self, test_function: Callable | None = None, reliability: float = 0.05
    ):
        super().__init__()
        self.test_function = test_function
        self.reliability = reliability
        self.default_kwargs: dict[str, Any] = {}

    @staticmethod  # TODO: remove
    def check_other(other: Dataset | None) -> Dataset:
        if other is None:
            raise ValueError("No other dataset provided")
        return other

    @staticmethod
    def check_dataset(data: Dataset):
        if len(data.columns) != 1:
            raise ValueError("Data must be one-dimensional")

    def check_data(self, data: Dataset, other: Dataset | None) -> Dataset:
        other = self.check_other(other)

        self.check_dataset(data)
        self.check_dataset(other)

        return other

    def _extract_arrays(self, data: Dataset, other: Dataset) -> tuple[Sequence]:
        raise NotImplementedError("This method should be relized using backend-dependent mixin.")

    @staticmethod
    def _form_results(
        p_value: float | None, statistic: float | None, reliability: float
    ) -> SmallDataset:
        return SmallDataset.from_dict({
            "p-value": p_value,
            "statistic": statistic,
            "pass": p_value < reliability,
        }, StatisticRole())

    def calc(
        self, data: Dataset, other: Dataset | None = None, **kwargs
    ) -> SmallDataset | float:
        other = self.check_data(data, other)
        if self.test_function is None:
            raise ValueError("test_function is needed for execution")
        
        import inspect
        sig = inspect.signature(self.test_function)
        valid_params = set(sig.parameters.keys())
        
        merged_kwargs = {**self.default_kwargs, **kwargs}
        
        invalid_keys = set(merged_kwargs.keys()) - valid_params
        if invalid_keys:
            merged_kwargs = {k: v for k, v in merged_kwargs.items() if k in valid_params}
        
        res = self.test_function(
            data._to_numpy(), other._to_numpy(), **merged_kwargs
        )
        return self._form_results(res[1], res[0], self.reliability)

class GroupTTestExtension(GroupStatTest):
    """
    Master-backend class for statistic test calculation.
    """
    test_function = staticmethod(ttest_ind)
    def __init__(self, reliability: float = 0.05):
        super().__init__(self.test_function, reliability)
        self.default_kwargs = {"nan_policy": "omit", "equal_var": False}

class GroupKSTestExtension(GroupStatTest):
    """
    Master-backend class for statistic test calculation.
    """
    test_function = staticmethod(ks_2samp)
    def __init__(self, reliability: float = 0.05):
        super().__init__(self.test_function, reliability)
        self.default_kwargs = {}

class GroupUTestExtension(GroupStatTest):
    """
    Master-backend class for statistic test calculation.
    """
    test_function = staticmethod(mannwhitneyu)
    def __init__(self, reliability: float = 0.05):
        super().__init__(self.test_function, reliability)
        self.default_kwargs = {"nan_policy": "omit"}

class GroupChi2TestExtension(GroupStatTest):
    """
    Master-backend class for statistic test calculation.
    """
    test_function = staticmethod(chi2_contingency)
    def __init__(self, reliability=0.05): super().__init__(self.test_function, reliability)

    def matrix_preparation(self, data: Dataset, other: Dataset) -> Dataset | None:
        raise NotImplementedError

    def calc(self, data, other = None, **kwargs):
        other = self.check_data(data, other)
        contingency_table = self.matrix_preparation(data, other)
        if contingency_table is None:
            warnings.warn(f"Matrix Chi2 is empty for {data.columns[0]}. Returning None")
            return DatasetAdapter.to_dataset(
                {
                    "p-value": [None],
                    "statistic": [None],
                    "pass": [None],
                },
                StatisticRole(),
            )
        statistic, p_value, *_ = chi2_contingency(contingency_table, **kwargs)
        return self._form_results(statistic, p_value, self.reliability)

@backend_factory.register(GroupKSTestExtension, PandasDataset)
class PandasKSTestExtension(GroupKSTestExtension):
    """
    Slave-backend class for statistical test calculation.
    """

@backend_factory.register(GroupChi2TestExtension, PandasDataset)
class PandasChi2TestExtension(GroupChi2TestExtension):
    """
    Slave-backend class for statistical test calculation.
    """

    @staticmethod
    def mini_category_replace(counts: Dataset) -> Dataset:
        mini_counts = counts["count"][counts["count"] < 7]
        if len(mini_counts) > 0:
            counts = counts.append(
                Dataset.from_dict(
                    [{counts.columns[0]: "other", "count": mini_counts["count"].sum()}],
                    roles=mini_counts.roles,
                )
            )
            counts = counts[counts["count"] >= 7]
        return counts

    def matrix_preparation(self, data: Dataset, other: Dataset) -> Dataset | None:
        proportion = len(data) / (len(data) + len(other))
        counted_data = data.value_counts()
        counted_data = self.mini_category_replace(counted_data)
        data_vc = counted_data["count"] * (1 - proportion)

        counted_other = other.value_counts()
        counted_other = self.mini_category_replace(counted_other)
        other_vc = counted_other["count"] * proportion

        if len(counted_data) < 2:
            return None
        data_vc = data_vc.add_column(counted_data[counted_data.columns[0]])
        other_vc = other_vc.add_column(counted_data[counted_data.columns[0]])
        return data_vc.merge(other_vc, on=counted_data.columns[0])[
            ["count_x", "count_y"]
        ].fillna(0)

@backend_factory.register(GroupKSTestExtension, SparkDataset)
class SparkKSTestExtension(GroupKSTestExtension):
    """
    Slave-backend class for statistical test calculation.
    """
    def __init__(self, reliability = 0.05, n_bins: int = 1000):
        super().__init__(reliability)
        self.n_bins = n_bins

    def calc(self, data: Dataset, other: Dataset | None = None, **kwargs) -> SmallDataset | float:
        """
        Compute the two-sample Kolmogorov-Smirnov (KS) test for PySpark-backed datasets.
        
        This method approximates the continuous KS test by discretizing the data into 
        a fixed number of histogram bins (`self.n_bins`) across the global range of 
        both datasets. It then computes the Empirical Cumulative Distribution Functions 
        (ECDFs) for both groups and finds the maximum absolute difference (D-statistic).
        Finally, it calculates the p-value using the asymptotic Kolmogorov distribution 
        (`kstwobign`) with Stephens' correction for finite sample sizes.

        Args:
            data (Dataset): The baseline dataset. Must contain exactly one numeric column.
            other (Dataset | None): The comparison dataset. Must contain exactly one numeric column.
            **kwargs: Additional keyword arguments (ignored in this implementation).

        Returns:
            Dataset | float: A SmallDataset containing the following fields:
                - 'p-value': The calculated p-value (float or None).
                - 'statistic': The KS D-statistic (float or None).
                - 'pass': Boolean flag indicating if p-value < self.reliability.
        """
        
        def _add_bucket_column(df: SparkDF, 
                               global_min: int, 
                               global_max: int) -> SparkDF:
            """
            Helper function to assign each row to a discrete histogram bin.
            Uses the global min/max to ensure both datasets share the exact same bin edges.
            """
            width = (global_max - global_min) / self.n_bins
            return df.withColumn(
                "bucket",
                F.least(
                    F.floor((F.col(col) - global_min) / width),
                    F.lit(self.n_bins - 1)  # Cap at the last bin to handle the max value edge case
                ).cast("int")
            )

        # Validate inputs and ensure both datasets are single-column and compatible
        other = self.check_data(data, other)
        df1 = data.data.to_spark()
        df2 = other.data.to_spark()
        col = data.columns[0]

        # Get sample sizes
        n1 = df1.count()
        n2 = df2.count()
        
        # Edge case: one or both datasets are empty
        if n1 == 0 or n2 == 0:
            return SmallDataset.from_dict({
                "p-value": None, "statistic": None, "pass": None
            }, StatisticRole())

        # Compute global minimum and maximum across both datasets to define common bin edges
        bounds1 = df1.agg(F.min(col).alias("min1"), F.max(col).alias("max1")).collect()[0]
        bounds2 = df2.agg(F.min(col).alias("min2"), F.max(col).alias("max2")).collect()[0]

        global_min = min(bounds1["min1"], bounds2["min2"])
        global_max = max(bounds1["max1"], bounds2["max2"])

        # Edge case: all values in both datasets are identical (zero variance)
        if global_min == global_max:
            return SmallDataset.from_dict({
                "p-value": 1.0,
                "statistic": 0.0,
                "pass": 1.0 < self.reliability
            }, StatisticRole())

        # Compute histograms (frequency counts per bin) for both datasets
        hist1 = (_add_bucket_column(df=df1, global_min=global_min, global_max=global_max)
            .groupBy("bucket")
            .count()
            .withColumnRenamed("count", "c1")
        )
        hist2 = (_add_bucket_column(df=df2, global_min=global_min, global_max=global_max)
            .groupBy("bucket")
            .count()
            .withColumnRenamed("count", "c2")
        )

        # Outer join histograms to align bins, filling missing bins with 0 counts
        combined = hist1.join(hist2, on="bucket", how="outer").fillna(0, subset=["c1", "c2"])
        
        # Convert to Pandas for efficient cumulative sum (ECDF calculation).
        # Note: The number of rows here is at most `self.n_bins`, so it safely fits in driver memory.
        pdf = combined.toPandas().sort_values("bucket").reset_index(drop=True)
        pdf["cum1"] = pdf["c1"].cumsum()
        pdf["cum2"] = pdf["c2"].cumsum()
        
        # Calculate the KS statistic: maximum absolute difference between the two ECDFs
        d_stat = float((pdf["cum1"]/n1 - pdf["cum2"]/n2).abs().max())

        try:
            # Calculate effective sample size for the asymptotic distribution
            en = np.sqrt(n1 * n2 / (n1 + n2))
            
            # Apply Stephens' correction (1970) for better accuracy with finite samples.
            # This is the exact formula used internally by scipy.stats.ks_2samp for large samples.
            p_value = float(kstwobign.sf((en + 0.12 + 0.11 / en) * d_stat))
        except Exception:
            # Fallback to 0.0 if the statistical function fails (e.g., due to extreme values)
            p_value = 0.0

        # Return the results as a standardized SmallDataset
        return SmallDataset.from_dict({
            "p-value": p_value,
            "statistic": d_stat,
            "pass": p_value < self.reliability
        }, StatisticRole())

@backend_factory.register(GroupChi2TestExtension, SparkDataset)
class SparkChi2TestExtension(GroupChi2TestExtension):
    """
    Slave-backend class for statistical test calculation.
    """

    @staticmethod
    def matrix_preparation(data: Dataset, other: Dataset) -> Dataset | None:
        other = np.array((
                            other
                            .data
                            .to_spark()
                            .rdd
                            .flatMap(lambda row: row)
                            .collect()
                        ))
        data = np.array((
                            data
                            .data
                            .to_spark()
                            .rdd
                            .flatMap(lambda row: row)
                            .collect()
                        ))
        unique_values = (set(other) | set(data))
        contingency_table = np.zeros((2, len(unique_values)))
        for index, element in enumerate(unique_values):
            contingency_table[0, index] = len(data[data == element])
            contingency_table[1, index] = len(other[other == element])

        return contingency_table

class NormCDF(GroupStatTest):
    def calc(
        self, data: Dataset, other: Dataset | None = None, **kwargs
    ) -> Dataset | float:
        result = norm.cdf(abs(data.get_values()[0][0]))
        return DatasetAdapter.to_dataset(
            {"p-value": 2 * (1 - result)},
            StatisticRole(),
        )
