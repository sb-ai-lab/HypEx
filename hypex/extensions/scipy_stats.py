from __future__ import annotations

import warnings
import numpy as np
from typing import Callable, Sequence, Any

from scipy.stats import (  # type: ignore
    chi2_contingency,
    ks_2samp,
    mannwhitneyu,
    norm,
    ttest_ind,
)
from ..utils.registry import backend_factory

from ..dataset import SmallDataset, Dataset, DatasetAdapter, StatisticRole
from ..dataset.backends import PandasDataset, SparkDataset
from .abstract import CompareExtension

class PandasExtractorMixin:
    """
    Pandas mixin for data extraction.
    """
    def _extract_arrays(self, data: Dataset, other: Dataset) -> tuple[Sequence]:
        return data.data.values.flatten(), other.data.values.flatten() 

class SparkExtractorMixin:
    """
    Spark mixin for data extraction.
    """
    def _extract_arrays(self, data: Dataset, other: Dataset) -> tuple[Sequence]:
        return (
            data.data.to_spark().rdd.flatMap(lambda row: row).collect(), 
            other.data.to_spark().rdd.flatMap(lambda row: row).collect()
        )

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
        data, other = self._extract_arrays(data, other)
        res = self.test_function(
            data, other, **self.default_kwargs, **kwargs
        )
        return self._form_results(res[1], res[0], self.reliability)

class GroupTTestExtension(GroupStatTest):
    """
    Master-backend class for statistic test calculation.
    """
    test_function = staticmethod(ttest_ind)
    default_kwargs = {"nan_policy": "omit", "equal_var": False}
    def __init__(self, reliability: float = 0.05): super().__init__(self.test_function, reliability)

class GroupKSTestExtension(GroupStatTest):
    """
    Master-backend class for statistic test calculation.
    """
    test_function = staticmethod(ks_2samp)
    default_kwargs = {}
    def __init__(self, reliability: float = 0.05): super().__init__(self.test_function, reliability)

class GroupUTestExtension(GroupStatTest):
    """
    Master-backend class for statistic test calculation.
    """
    test_function = staticmethod(mannwhitneyu)
    default_kwargs = {"nan_policy": "omit"}
    def __init__(self, reliability: float = 0.05): super().__init__(self.test_function, reliability)

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
        statistic, p_value, _, _ = chi2_contingency(contingency_table, **kwargs)
        return self._form_results(statistic, p_value, self.reliability)

@backend_factory.register(GroupTTestExtension, PandasDataset)
class PandasTTestExtension(PandasExtractorMixin, GroupTTestExtension):
    """
    Slave-backend class for statistical test calculation.
    """

@backend_factory.register(GroupKSTestExtension, PandasDataset)
class PandasKSTestExtension(PandasExtractorMixin, GroupKSTestExtension):
    """
    Slave-backend class for statistical test calculation.
    """

@backend_factory.register(GroupUTestExtension, PandasDataset)
class PandasUTestExtension(PandasExtractorMixin, GroupUTestExtension):
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
    


@backend_factory.register(GroupTTestExtension, SparkDataset)
class SparkTTestExtension(SparkExtractorMixin, GroupTTestExtension):
    """
    Slave-backend class for statistical test calculation.
    """

@backend_factory.register(GroupKSTestExtension, SparkDataset)
class SparkKSTestExtension(SparkExtractorMixin, GroupKSTestExtension):
    """
    Slave-backend class for statistical test calculation.
    """

@backend_factory.register(GroupUTestExtension, SparkDataset)
class SparkUTestExtension(SparkExtractorMixin, GroupUTestExtension):
    """
    Slave-backend class for statistical test calculation.
    """

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
    def _calc_pandas(
        self, data: Dataset, other: Dataset | None = None, **kwargs
    ) -> Dataset | float:
        result = norm.cdf(abs(data.get_values()[0][0]))
        return DatasetAdapter.to_dataset(
            {"p-value": 2 * (1 - result)},
            StatisticRole(),
        )
