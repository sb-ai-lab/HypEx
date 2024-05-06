from typing import Callable

from scipy.stats import ttest_ind, ks_2samp, chi2_contingency

from typing import Union, List, Callable

import pandas as pd

from hypex.dataset import Dataset, StatisticRole
from hypex.dataset.tasks.abstract import CompareTask, Task
from hypex.utils import BackendsEnum


class StatTest(CompareTask):
    def __init__(self, test_function: Callable = None, alpha: float = 0.05):
        super().__init__()
        self.test_function = test_function
        self.alpha = alpha

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
                    "pass": one_result.pvalue < self.alpha,
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
        return self.convert_scipy_to_dataset(one_result)


class TTest(StatTest):
    def __init__(self, alpha: float = 0.05):
        super().__init__(ttest_ind, alpha=alpha)


class KSTest(StatTest):
    def __init__(self, alpha: float = 0.05):
        super().__init__(ks_2samp, alpha=alpha)


class Chi2Test(StatTest):
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
