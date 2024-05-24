from typing import Callable, Union, Optional

import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu, ttest_ind

from hypex.dataset import Dataset, StatisticRole
from hypex.extensions.abstract import CompareExtension
from hypex.utils import BackendsEnum


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

    def convert_scipy_to_dataset(self, one_result):
        df_result = pd.DataFrame(
            [
                {
                    "p-value": one_result.pvalue,
                    "statistic": one_result.statistic,
                    "pass": one_result.pvalue < self.reliability,
                }
            ]
        )
        return Dataset(
            data=df_result,
            backend=BackendsEnum.pandas,
            roles={str(f): StatisticRole() for f in df_result.columns},
        )

    def _calc_pandas(
        self, data: Dataset, other: Union[Dataset, None] = None, **kwargs
    ) -> Union[float, Dataset]:
        other = self.check_data(data, other)
        one_result = self.test_function(
            data.backend.data.values.flatten(), other.backend.data.values.flatten()
        )
        one_result = self.convert_scipy_to_dataset(one_result)
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
        data_vc = data.value_counts() * (1 - proportion)
        other_vc = other.value_counts() * proportion

        return data_vc.merge(other_vc).fillna(0)

    def _calc_pandas(
        self, data: Dataset, other: Union[Dataset, None] = None, **kwargs
    ) -> Union[float, Dataset]:
        other = self.check_data(data, other)
        matrix = self.matrix_preparation(data, other)
        one_result = chi2_contingency(matrix.backend.data)
        return self.convert_scipy_to_dataset(one_result)
