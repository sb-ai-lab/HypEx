from typing import Optional, Union, List, Dict

import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests  # type: ignore

from hypex.dataset import Dataset, StatisticRole
from hypex.utils import ABNTestMethodsEnum
from .abstract import Extension


class ABMultiTest(Extension):
    def __init__(self, method: ABNTestMethodsEnum, alpha: float = 0.05):
        self.method = method
        self.alpha = alpha
        super().__init__()

    @staticmethod
    def multitest_result_to_dataset(result: Dict):
        return Dataset.from_dict(
            result, roles={column: StatisticRole() for column in result.keys()}
        )

    def _calc_pandas(self, data: Dataset, **kwargs):
        p_values = data.data.values.flatten()
        result = multipletests(
            p_values, method=self.method.value, alpha=self.alpha, **kwargs
        )
        return self.multitest_result_to_dataset(
            {"rejected": result[0], "new p-values": result[1]}
        )


class ABMultitestQuantile(Extension):
    def __init__(
        self,
        alpha: float = 0.05,
        iteration_size: int = 20000,
        equal_variance: bool = True,
        random_state: Optional[int] = None,
    ):
        self.alpha = alpha
        self.iteration_size = iteration_size
        self.equal_variance = equal_variance
        self.random_state = random_state
        super().__init__()

    @staticmethod
    def multitest_result_to_dataset(result: Dict):
        return Dataset.from_dict(
            result, roles={column: StatisticRole() for column in result.keys()}
        )

    def _calc_pandas(self, data: Dataset, **kwargs):
        group_field = kwargs.get("group_field")
        target_field = kwargs.get("target_field")
        quantiles = kwargs.get("quantiles")
        num_samples = len(data.unique()[group_field])
        sample_size = len(data)
        grouped_data = data.groupby(by=group_field, fields_list=target_field)
        means = [sample[1].agg("mean") for sample in grouped_data]
        variances = [
            sample[1].agg("var") * sample_size / (sample_size - 1)
            for sample in grouped_data
        ]
        if num_samples != len(means) or num_samples != len(variances):
            num_samples = min(num_samples, len(means), len(variances))
        if type(quantiles) is float:
            quantiles = np.full(num_samples, quantiles).tolist()

        quantiles = quantiles or self.quantile_of_marginal_distribution(
            num_samples=num_samples,
            quantile_level=1 - self.alpha / num_samples,
            variances=variances,
        )
        for j in range(num_samples):
            min_t_value = np.inf
            for i in range(num_samples):
                if i != j:
                    t_value = (
                        np.sqrt(sample_size)
                        * (means[j] - means[i])
                        / np.sqrt(variances[j] + variances[i])
                    )
                    min_t_value = min(min_t_value, t_value)
            if min_t_value > quantiles[j]:
                return self.multitest_result_to_dataset(
                    {"accepted hypothesis": [j + 1]}
                )
        return self.multitest_result_to_dataset({"accepted hypothesis": [0]})

    def quantile_of_marginal_distribution(
        self,
        num_samples: int,
        quantile_level: float,
        variances: Optional[List[float]] = None,
    ) -> List[float]:
        if variances is None:
            self.equal_variance = True
        num_samples_hyp = 1 if self.equal_variance else num_samples
        quantiles = []
        for j in range(num_samples_hyp):
            t_values = []
            random_samples = norm.rvs(
                size=[self.iteration_size, num_samples], random_state=self.random_state
            )
            for sample in random_samples:
                min_t_value = np.inf
                for i in range(num_samples):
                    if i != j:
                        if self.equal_variance:
                            t_value = (sample[j] - sample[i]) / np.sqrt(2)
                        else:
                            t_value = sample[j] / np.sqrt(
                                1 + variances[i] / variances[j]
                            ) - sample[i] / np.sqrt(1 + variances[j] / variances[i])
                        min_t_value = min(min_t_value, t_value)
                t_values.append(min_t_value)
            quantiles.append(np.quantile(t_values, quantile_level))
        return (
            np.full(num_samples, quantiles[0]).tolist()
            if self.equal_variance
            else quantiles
        )

    def min_sample_size(
        self,
        num_samples: int,
        mde: float,
        variances: Union[List[float], float],
        power: float = 0.2,
        quantile_1: Optional[Union[float, List[float]]] = None,
        quantile_2: Optional[Union[float, List[float]]] = None,
        initial_estimate: int = 0,
        iteration_size: int = 3000,
    ):
        if type(quantile_1) is float:
            quantile_1 = np.full(num_samples, quantile_1).tolist()
        if type(quantile_2) is float:
            quantile_2 = np.full(num_samples, quantile_2).tolist()

        quantile_1 = quantile_1 or self.quantile_of_marginal_distribution(
            num_samples=num_samples,
            quantile_level=1 - self.alpha / num_samples,
            variances=variances,
        )
        quantile_2 = quantile_2 or self.quantile_of_marginal_distribution(
            num_samples=num_samples, quantile_level=power
        )

        if self.equal_variance:
            return int(2 * variances * ((quantile_1[0] - quantile_2[0]) / mde) ** 2) + 1
        else:
            sizes = []
            for index in range(num_samples):
                size = initial_estimate
                current_power = 0
                while current_power < 1 - power:
                    size += 100
                    current_power = 0
                    total_samples = norm.rvs(
                        size=[iteration_size, num_samples],
                        random_state=self.random_state,
                    )
                    for sample in total_samples:
                        min_t_value = np.inf
                        for i in range(num_samples):
                            if i != index:
                                t_value = (
                                    sample[index]
                                    / np.sqrt(1 + variances[i] / variances[index])
                                    - sample[i]
                                    / np.sqrt(1 + variances[index] / variances[i])
                                    + mde
                                    * np.sqrt(size / (variances[index] + variances[i]))
                                )
                                min_t_value = min(min_t_value, t_value)
                        if min_t_value > quantile_1[index]:
                            current_power += 1
                    current_power /= iteration_size
                sizes.append(size)
            return {"min sample size": np.max(sizes)}