from typing import Callable

from scipy.stats import ttest_ind, ks_2samp

from typing import Union, List, Callable

import pandas as pd

from hypex.dataset import Dataset, StatisticRole
from hypex.dataset.tasks.abstract import CompareTask
from hypex.utils import BackendsEnum


class StatTest(CompareTask):
    def __init__(self, test_function: Callable, alpha: float = 0.05):
        self.test_function = test_function
        self.alpha = alpha

    @staticmethod
    def check_other(other: Union[Dataset, List[Dataset]]):
        if len(other) == 0:
            raise ValueError("No other dataset provided")

    @staticmethod
    def check_dataset(data: Dataset):
        if len(data.columns) != 1:
            raise ValueError("Data must be one-dimensional")

    def calc(self, data: Dataset, other: Union[Dataset, List[Dataset], None] = None, **kwargs) -> Union[float, Dataset]:
        other = other or []
        self.check_other(other)

        if isinstance(other, Dataset):
            other = [other]

        self.check_dataset(data)

        result = []
        for o in other:
            self.check_dataset(o)
            one_result = self.test_function(data.data.values.flatten(), o.data.values.flatten())
            result.append(
                {
                    "p-value": one_result.pvalue,
                    "statistic": one_result.statistic,
                    "pass": one_result.pvalue < self.alpha,
                }
            )

        df_result = pd.DataFrame(result)
        return Dataset(
            roles={str(f): StatisticRole() for f in df_result.columns},
            data=df_result,
            backend=BackendsEnum.pandas
        )


class TTest(StatTest):
    def __init__(self, alpha: float = 0.05):
        super().__init__(ttest_ind, alpha=alpha)


class KSTest(StatTest):
    def __init__(self, alpha: float = 0.05):
        super().__init__(ks_2samp, alpha=alpha)
