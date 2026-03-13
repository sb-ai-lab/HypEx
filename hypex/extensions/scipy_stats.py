from __future__ import annotations

import warnings
import numpy as np
from typing import Callable

from scipy.stats import (  # type: ignore
    chi2_contingency,
    ks_2samp,
    mannwhitneyu,
    norm,
    ttest_ind,
)

from ..dataset import SmallDataset, Dataset, DatasetAdapter, StatisticRole
from ..dataset.backends import PandasDataset, SparkDataset
from .abstract import CompareExtension


class StatTest(CompareExtension):
    def __init__(
        self, test_function: Callable | None = None, reliability: float = 0.05
    ):
        super().__init__()
        self.test_function = test_function
        self.reliability = reliability

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

    def _calc_pandas(
        self, data: Dataset, other: Dataset | None = None, **kwargs
    ) -> Dataset | float:
        other = self.check_data(data, other)
        if self.test_function is None:
            raise ValueError("test_function is needed for execution")
        one_result = self.test_function(
            data.backend.data.values.flatten(),
            other.backend.data.values.flatten(),
            **kwargs,
        )
        one_result = SmallDataset.from_dict(
            {
                "p-value": one_result.pvalue,
                "statistic": one_result.statistic,
                "pass": one_result.pvalue < self.reliability,
            },
            StatisticRole(),
        )
        return one_result
    
    def _calc_spark(
        self, data: Dataset, other: Dataset | None = None, **kwargs
        ) -> Dataset | float:
        other = self.check_data(data, other)
        if self.test_function is None:
            raise ValueError("test_function is needed for execution")
        one_result = self.test_function(
            data.data.to_spark().rdd.flatMap(lambda row: row).collect(),
            other.data.to_spark().rdd.flatMap(lambda row: row).collect(),
            **kwargs
        )
        one_result = SmallDataset.from_dict(
            {
                "p-value": one_result[1],
                "statistic": one_result[0],
                "pass": one_result[1] < self.reliability,
            },
            StatisticRole(),
        )
        return one_result


class TTestExtension(StatTest):
    def __init__(self, reliability: float = 0.05):
        super().__init__(ttest_ind, reliability=reliability)

    def _calc_pandas(
        self, data: Dataset, other: Dataset | None = None, **kwargs
    ) -> Dataset | float:
        # if (
        #     next(iter(data.nunique().values()))
        #     and next(iter(other.nunique().values())) < 2
        # ):
        #     return DatasetAdapter.to_dataset(
        #         {
        #             "p-value": [None],
        #             "statistic": [None],
        #             "pass": [None],
        #         },
        #         StatisticRole(),
        #     )
        return super()._calc_pandas(data, other, nan_policy="omit", **kwargs)


class KSTestExtension(StatTest):
    def __init__(self, reliability: float = 0.05):
        super().__init__(ks_2samp, reliability=reliability)


class UTestExtension(StatTest):
    def __init__(self, reliability: float = 0.05):
        super().__init__(mannwhitneyu, reliability=reliability)


class Chi2TestExtension(StatTest):
    def __init__(self, test_function = None, reliability = 0.05):
        super().__init__(test_function, reliability)
        self.DATA_MAPPER = {
            PandasDataset: self._pandas_prep,
            SparkDataset: self._spark_prep
        }

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
        return self.DATA_MAPPER[type(data.backend)](data, other)
    
    def _pandas_prep(self, data: Dataset, other: Dataset) -> Dataset | None:
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
    
    @staticmethod
    def _spark_prep(data: Dataset, other: Dataset) -> np.ndarray:
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

    def _calc_pandas(
        self, data: Dataset, other: Dataset | None = None, **kwargs
    ) -> Dataset | float:
        other = self.check_data(data, other)
        matrix = self.matrix_preparation(data, other)
        if matrix is None:
            warnings.warn(f"Matrix Chi2 is empty for {data.columns[0]}. Returning None")
            return DatasetAdapter.to_dataset(
                {
                    "p-value": [None],
                    "statistic": [None],
                    "pass": [None],
                },
                StatisticRole(),
            )
        one_result = chi2_contingency(matrix.backend.data)
        return DatasetAdapter.to_dataset(
            {
                "p-value": (
                    one_result[1]
                    if isinstance(one_result, tuple)
                    else one_result.pvalue
                ),
                "statistic": (
                    one_result[0]
                    if isinstance(one_result, tuple)
                    else one_result.statistic
                ),
                "pass": (
                    one_result[1]
                    if isinstance(one_result, tuple)
                    else one_result.pvalue
                )
                < self.reliability,
            },
            StatisticRole(),
        )
    
    def _calc_spark(self, data, other = None, **kwargs):
        other = self.check_data(data, other)
        contingency_table = self.matrix_preparation(data, other)

        statistic, pvalue, dof, expected_freq = chi2_contingency(contingency_table, **kwargs)
        one_result = SmallDataset.from_dict(
            {
                "p-value": pvalue,
                "statistic": statistic,
                "pass": pvalue < self.reliability,
            },
            StatisticRole(),
        )
        return one_result


class NormCDF(StatTest):
    def _calc_pandas(
        self, data: Dataset, other: Dataset | None = None, **kwargs
    ) -> Dataset | float:
        result = norm.cdf(abs(data.get_values()[0][0]))
        return DatasetAdapter.to_dataset(
            {"p-value": 2 * (1 - result)},
            StatisticRole(),
        )
