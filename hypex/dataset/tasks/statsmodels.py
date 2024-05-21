from typing import Optional, Union, List, Dict

import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests  # type: ignore

from hypex.utils import ABNTestMethodsEnum
from .abstract import Task
from .. import Dataset, StatisticRole


class ABMultiTest(Task):
    def __init__(
        self,
        method: ABNTestMethodsEnum,
        alpha: float = 0.05,
        random_state: Optional[int] = None,
    ):
        self.method = method
        self.alpha = alpha
        self.random_state = random_state
        super().__init__()

    @staticmethod
    def multitest_result_to_dataset(result: Dict):
        return Dataset.from_dict(
            result, roles={column: StatisticRole() for column in result.keys()}
        )

    def _calc_pandas(self, data: Dataset, **kwargs):
        if self.method.value == "quantile":
            return self.multitest_result_to_dataset(
                self.test_on_marginal_distribution(
                    data,
                    kwargs.get("group_field"),
                    kwargs.get("target_field"),
                    kwargs.get("equal_variance", True),
                    kwargs.get("quantiles", None),
                )
            )
        p_values = data.data.values.flatten()
        result = multipletests(
            p_values, method=self.method.value, alpha=self.alpha, **kwargs
        )
        return self.multitest_result_to_dataset(
            {"rejected": result[0], "new p-values": result[1]}
        )

    def test_on_marginal_distribution(
        self,
        samples: Dataset,
        group_field: str,
        target_field: str,
        equal_variance: bool = True,
        quantiles: Optional[Union[float, List[float]]] = None,
    ):
        num_samples = len(samples)
        sample_size = len(samples[0])
        grouped_data = samples.groupby(by=group_field, fields_list=target_field)
        means = [sample[1].agg("mean") for sample in grouped_data]
        variances = [
            sample[1].agg("var") * sample_size / (sample_size - 1)
            for sample in grouped_data
        ]
        if type(quantiles) is float:
            quantiles = np.full(num_samples, quantiles).tolist()

        quantiles = quantiles or self.quantile_of_marginal_distribution(
            num_samples=num_samples,
            quantile_level=1 - self.alpha / num_samples,
            variances=variances,
            equal_variance=equal_variance,
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
                return {"accepted hypothesis": j + 1}
        return {"accepted hypothesis": 0}

    def quantile_of_marginal_distribution(
        self,
        num_samples: int,
        quantile_level: float,
        variances: Optional[List[float]] = None,
        equal_variance: bool = True,
        iteration_size: int = 20000,
    ) -> List[float]:
        if variances is None:
            equal_variance = True
        num_samples_hyp = 1 if equal_variance else num_samples
        quantiles = []
        for j in range(num_samples_hyp):
            t_values = []
            random_samples = norm.rvs(
                size=[iteration_size, num_samples], random_state=self.random_state
            )
            for sample in random_samples:
                min_t_value = np.inf
                for i in range(num_samples):
                    if i != j:
                        if equal_variance:
                            t_value = (sample[j] - sample[i]) / np.sqrt(2)
                        else:
                            t_value = sample[j] / np.sqrt(
                                1 + variances[i] / variances[j]
                            ) - sample[i] / np.sqrt(1 + variances[j] / variances[i])
                        min_t_value = min(min_t_value, t_value)
                t_values.append(min_t_value)
            quantiles.append(np.quantile(t_values, quantile_level))
        return (
            np.full(num_samples, quantiles[0]).tolist() if equal_variance else quantiles
        )

    def min_sample_size(
        self,
        number_of_samples: int,
        minimum_detectable_effect: float,
        variances: Union[List[float], float],
        power_level: Optional[float] = 0.2,
        equal_variance: Optional[bool] = True,
        quantile_1: Optional[Union[float, List[float]]] = None,
        quantile_2: Optional[Union[float, List[float]]] = None,
        initial_estimate: Optional[int] = None,
        iteration_size: Optional[int] = 3000,
    ):
        if type(quantile_1) is float:
            quantile_1 = np.full(number_of_samples, quantile_1).tolist()
        if type(quantile_2) is float:
            quantile_2 = np.full(number_of_samples, quantile_2).tolist()

        if quantile_1 is None:
            quantile_1 = self.quantile_of_marginal_distribution(
                num_samples=number_of_samples,
                quantile_level=1 - self.alpha / number_of_samples,
                variances=variances,
                equal_variance=equal_variance,
            )
        if quantile_2 is None:
            quantile_2 = self.quantile_of_marginal_distribution(
                num_samples=number_of_samples, quantile_level=power_level
            )

        if equal_variance:
            return (
                int(
                    2
                    * variances
                    * ((quantile_1[0] - quantile_2[0]) / minimum_detectable_effect) ** 2
                )
                + 1
            )
        else:
            sample_sizes = []
            for sample_index in range(number_of_samples):
                sample_size = initial_estimate or 0
                current_power = 0
                while current_power < 1 - power_level:
                    sample_size += 100
                    current_power = 0
                    total_samples = norm.rvs(
                        size=[iteration_size, number_of_samples],
                        random_state=self.random_state,
                    )
                    for sample in total_samples:
                        min_t_value = np.inf
                        for i in range(number_of_samples):
                            if i != sample_index:
                                t_value = (
                                    sample[sample_index]
                                    / np.sqrt(
                                        1 + variances[i] / variances[sample_index]
                                    )
                                    - sample[i]
                                    / np.sqrt(
                                        1 + variances[sample_index] / variances[i]
                                    )
                                    + minimum_detectable_effect
                                    * np.sqrt(
                                        sample_size
                                        / (variances[sample_index] + variances[i])
                                    )
                                )
                                min_t_value = min(min_t_value, t_value)
                        if min_t_value > quantile_1[sample_index]:
                            current_power += 1
                    current_power /= iteration_size
                sample_sizes.append(sample_size)
            return {"min sample size": np.max(sample_sizes)}
