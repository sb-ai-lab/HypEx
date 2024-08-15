from typing import Callable, Union, Optional

from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu, ttest_ind  # type: ignore

from hypex.dataset import Dataset, StatisticRole, DatasetAdapter
from hypex.extensions.abstract import CompareExtension


class StatTest(CompareExtension):
    def __init__(
        self, test_function: Optional[Callable] = None, reliability: float = 0.05
    ):
        super().__init__()
        self.test_function = test_function
        self.reliability = reliability

    @staticmethod
    def check_other(other: Union[Dataset, None]) -> Dataset:
        if other is None:
            raise ValueError("No other dataset provided")
        return other

    @staticmethod
    def check_dataset(data: Dataset):
        if len(data.columns) != 1:
            raise ValueError("Data must be one-dimensional")

    def check_data(self, data: Dataset, other: Union[Dataset, None]) -> Dataset:
        other = self.check_other(other)

        self.check_dataset(data)
        self.check_dataset(other)

        return other

    def _calc_pandas(
        self, data: Dataset, other: Union[Dataset, None] = None, **kwargs
    ) -> Union[float, Dataset]:
        other = self.check_data(data, other)
        if self.test_function is None:
            raise ValueError("test_function is needed for execution")
        one_result = self.test_function(
            data.backend.data.values.flatten(), other.backend.data.values.flatten()
        )
        one_result = DatasetAdapter.to_dataset(
            {
                "p-value": one_result.pvalue,
                "statistic": one_result.statistic,
                "pass": one_result.pvalue < self.reliability,
            },
            StatisticRole(),
        )
        return one_result


class TTestExtension(StatTest):
    def __init__(self, reliability: float = 0.05):
        super().__init__(ttest_ind, reliability=reliability)


class KSTestExtension(StatTest):
    def __init__(self, reliability: float = 0.05):
        super().__init__(ks_2samp, reliability=reliability)


class UTestExtension(StatTest):
    def __init__(self, reliability: float = 0.05):
        super().__init__(mannwhitneyu, reliability=reliability)


class Chi2TestExtension(StatTest):
    @staticmethod
    def matrix_preparation(data: Dataset, other: Dataset):
        proportion = len(data) / (len(data) + len(other))
        counted_data = data.value_counts()
        data_vc = counted_data["count"] * (1 - proportion)
        other_vc = other.value_counts()["count"] * proportion
        data_vc = data_vc.add_column(counted_data[counted_data.columns[0]])
        other_vc = other_vc.add_column(counted_data[counted_data.columns[0]])
        return data_vc.merge(other_vc, on=counted_data.columns[0])[
            ["count_x", "count_y"]
        ].fillna(0)

    def _calc_pandas(
        self, data: Dataset, other: Optional[Dataset] = None, **kwargs
    ) -> Union[float, Dataset]:
        other = self.check_data(data, other)
        matrix = self.matrix_preparation(data, other)
        one_result = chi2_contingency(matrix.backend.data)
        return DatasetAdapter.to_dataset(
            {
                "p-value": one_result.pvalue,
                "statistic": one_result.statistic,
                "pass": one_result.pvalue < self.reliability,
            },
            StatisticRole(),
        )
