"""
Multiple Hypothesis Testing for Best Sample Selection
=====================================================

This module outlines a statistical approach developed to select the best sample out of k samples based on a specific target metric. It is designed to adhere to predetermined Type I and Type II error probabilities, leveraging the marginal distribution of the minimum statistic. This documentation provides an overview of the problem, the proposed solution, and detailed descriptions of the main functions implemented.

Problem Statement:
------------------
In multiple hypothesis testing, our aim is to identify the superior sample from a set of k samples, each evaluated according to a certain target metric. We denote by H_i the hypothesis that the i-th sample is the best among all, for i = 1, ..., k, and by H_0 the null hypothesis that there is no single best sample among the group.

Methodology:
------------
The method is based on the limit distribution theory, allowing us to maintain the predefined probabilities of committing Type I and Type II errors. The core approach involves calculating the minimum sample size required for a reliable comparison and applying a criterion to test the hypotheses, both of which rely on the quantile function of the marginal distribution of the minimum statistic.

Functions Overview:
-------------------
1. Quantile of Marginal Distribution: Calculates the quantile of the marginal distribution of the minimum of a set of random variables. Essential for determining the critical values used in the main hypothesis testing and sample size calculation functions.

2. Minimum Sample Size Calculation: Determines the smallest number of observations required in each sample to reliably detect a specified effect size with a given power and significance level.

3. Hypothesis Testing Criterion: Tests whether any of the samples is significantly better than the others based on their target metric distributions.

Implementation:
---------------
Implemented in Python, utilizing numpy for numerical computations and scipy.stats for statistical functions. Detailed function descriptions, including input parameters and return values, are documented within the function docstrings in this file.

Mathematical Formulations:
--------------------------
The quantile calculation for the marginal distribution of the minimum statistic is based on the following principle:

Q(p; F_1, ..., F_k) = min { q_1(p), ..., q_k(p) }

where Q(p; F_1, ..., F_k) represents the p-th quantile of the marginal distribution of the minimum of k random variables with distribution functions F_1, ..., F_k, and q_i(p) denotes the p-th quantile of the i-th distribution.

Conclusion:
-----------
This module provides a robust solution for statistical comparison across multiple samples, ensuring adherence to specified error probabilities through the application of limit distribution theory.

"""
from typing import Optional, List, Union

import numpy as np
from scipy.stats import norm


def quantile_of_marginal_distribution(num_samples: int, quantile_level: float, variances: Optional[List, float] = None,
                                      iteration_size: int = 20000):
    """Calculate the quantile(s) of the marginal distribution for minimum t-values across multiple comparisons.

    This function generates random samples from a normal distribution and computes t-values for comparisons either
    with equal variances (if variances are not provided) or with specified unequal variances (if variances are provided).
    It then determines the quantile of interest for the distribution of the minimum t-values from these comparisons.

    Args:
        num_samples: The number of samples/groups to compare.
        quantile_level: The quantile level to compute for the marginal distribution (e.g., 0.95 for the 95th percentile).
        variances: A list of variances for each sample/group. If None, equal variances are assumed.
        iteration_size: The number of iterations/random samples to generate for the simulation.

    Returns:
       The quantile of interest for the marginal distribution of the minimum t-values. Returns
       a single float if variances are assumed equal (or not provided) or a list of floats
       with quantiles for each sample if variances are provided and unequal.
    """
    num_samples_hyp = num_samples if variances else 1

    quantiles = []
    for j in range(num_samples_hyp):
        t_values = []
        random_samples = norm.rvs(size=[iteration_size, num_samples], random_state=random_state)
        for sample in random_samples:
            min_t_value = np.inf
            for i in range(num_samples):
                if i != j:
                    if variances is None:
                        t_value = (sample[0] - sample[i]) / np.sqrt(2)
                    else:
                        t_value = sample[j] / np.sqrt(1 + variances[i] / variances[j]) - sample[i] / np.sqrt(
                            1 + variances[j] / variances[i])
                    min_t_value = min(min_t_value, t_value)
            t_values.append(min_t_value)
        quantiles.append(np.quantile(t_values, quantile_level))
    return quantiles if variances else quantiles[0]


def test_on_marginal_distribution(samples: List[np.ndarray], significance_level: float = 0.05,
                                  equal_variance: bool = True,
                                  quantiles: Optional[Union[float, List[float]]] = None) -> int:
    """Performs a test on the marginal distribution of minimum t-values across multiple samples/groups.

    This function calculates the means and variances for each sample/group, determines the quantile of interest for
    the marginal distribution of the minimum t-values, and identifies if any sample's minimum t-value exceeds this
    quantile. It is used to assess whether there is a statistically significant difference between any of the samples.

    Args:
        samples: A list of arrays, where each array represents the data of a sample/group.
        significance_level: The significance level for the test (default is 0.05).
        equal_variance: A boolean indicating if the samples are assumed to have equal variance (default is True).
        quantiles: Pre-computed quantiles of the marginal distribution. If None, they will be computed.

    Returns:
        The index of the first sample that significantly differs from others, or 0 if none are found.
    """
    num_samples = len(samples)
    sample_size = len(samples[0])

    means = [np.mean(sample) for sample in samples]
    variances_q = [np.var(sample) * sample_size / (sample_size - 1) for sample in samples] if equal_variance else None
    variances = [np.var(sample) * sample_size / (sample_size - 1) for sample in samples]

    if quantiles is None:
        quantiles = quantile_of_marginal_distribution(num_samples=num_samples,
                                                      quantile_level=1 - significance_level / num_samples,
                                                      variances=variances_q)
    for j in range(num_samples):
        min_t_value = np.inf
        for i in range(num_samples):
            if i != j:
                t_value = np.sqrt(sample_size) * (means[j] - means[i]) / np.sqrt(variances[j] + variances[i])
                min_t_value = min(min_t_value, t_value)

        if min_t_value > quantiles[j]:
            return j + 1
    return 0


def min_sample_size(number_of_samples: int, minimum_detectable_effect: float, variances: Union[List[float], float],
                    significance_level: float = 0.05, power_level: float = 0.2, equal_variance: bool = True,
                    quantile_1: Optional[Union[float, List[float]]] = None, quantile_2: Optional[float] = None,
                    initial_estimate: Optional[int] = None) -> int:
    """
    Calculates the minimum sample size required to detect a given effect with specified power and significance level.

    The function computes the minimum sample size for either equal or unequal variances across samples. It can use
    pre-computed quantiles of the marginal distribution to speed up calculations.

    Args:
        number_of_samples: Number of samples or groups to compare.
        minimum_detectable_effect: The minimum effect size that the test is designed to detect.
        variances: Variances of the samples. Can be a single value if equal_variance is True, or a list of variances.
        significance_level: The significance level for hypothesis testing (alpha).
        power_level: Desired power of the test (1 - beta).
        equal_variance: Boolean flag indicating whether to assume equal variances across samples.
        quantile_1: Optional pre-computed quantile for the significance level. Calculated if None.
        quantile_2: Optional pre-computed quantile for the power level. Calculated if None.
        initial_estimate: Optional initial estimate for the sample size to speed up calculations.

    Returns:
        The minimum sample size required per sample/group.
    """
    random_state = 42
    if equal_variance:

        if quantile_1 is None:
            quantile_1 = quantile_of_marginal_distribution(num_samples=number_of_samples,
                                                           quantile_level=1 - significance_level / number_of_samples)  # квантиль предельного распределения 1-alpha/k

        if quantile_2 is None:
            quantile_2 = quantile_of_marginal_distribution(num_samples=number_of_samples,
                                                           quantile_level=power_level)  # квантиль предельного распределения beta

        print(f"{quantile_1 = }, {quantile_2 = }")

        return int(2 * variances * ((quantile_1 - quantile_2) / minimum_detectable_effect) ** 2) + 1
    else:
        iteration_size = 3000  # Количество итераций
        if quantile_1 is None:
            quantile_1 = quantile_of_marginal_distribution(num_samples=number_of_samples,
                                                           quantile_level=1 - significance_level / number_of_samples,
                                                           variances=variances,
                                                           )  # набор квантилей предельного распределения
        sample_sizes = []  # для размеров выборки
        for j in range(number_of_samples):
            sample_size = initial_estimate or 0
            current_power = 0  # мощность
            while current_power < 1 - power_level:
                sample_size += 100
                current_power = 0
                total_samples = norm.rvs(size=[iteration_size, number_of_samples], random_state=random_state)
                for sample in total_samples:
                    min_t_value = np.inf
                    for i in range(number_of_samples):
                        if i != j:
                            t_value = sample[j] / np.sqrt(1 + variances[i] / variances[j]) - sample[i] / np.sqrt(
                                1 + variances[j] / variances[i]) + minimum_detectable_effect * np.sqrt(
                                sample_size / (variances[j] + variances[i]))
                            min_t_value = min(min_t_value, t_value)
                    if min_t_value > quantile_1[j]:
                        current_power += 1
                current_power /= iteration_size
            sample_sizes.append(sample_size)
        return np.max(sample_sizes)


# Применение метода
import numpy as np
from scipy.stats import bernoulli

# Initialize random state
seed = 42  # You can choose any number as the seed
random_state = np.random.RandomState(seed)

# Multiple testing for best sample selection
# Number of samples and parameters
num_samples = 10  # Number of samples
minimum_detectable_effect = 0.05  # MDE
assumed_conversion = 0.3  # Assumed conversion rate
significance_level = 0.05  # Significance level
power_level = 0.2  # Power level (1 - beta)

# Calculate the minimum sample size
sample_size = min_sample_size(num_samples, minimum_detectable_effect,
                              variances=assumed_conversion * (1 - assumed_conversion),
                              significance_level=significance_level, power_level=power_level, equal_variance=True)
print(f'Sample size = {sample_size}')

# Testing samples with equal conversion rate
print('\nSamples with equal conversion rate')
for _ in range(5):
    samples = bernoulli.rvs(assumed_conversion, size=[num_samples, sample_size], random_state=random_state)
    hypothesis = test_on_marginal_distribution(samples, significance_level=significance_level)
    print(f'\tAccepted hypothesis H({hypothesis})')

print("kek")

# Testing where the last sample has a higher conversion rate by MDE
print('\nLast sample has higher conversion by MDE')
for _ in range(5):
    samples = [bernoulli.rvs(assumed_conversion, size=sample_size, random_state=random_state) for _ in
               range(num_samples - 1)]
    samples.append(
        bernoulli.rvs(assumed_conversion + minimum_detectable_effect, size=sample_size, random_state=random_state))
    hypothesis = test_on_marginal_distribution(samples, significance_level=significance_level)
    print(f'\tAccepted hypothesis H({hypothesis})')

# Multiple testing for best client income sample (conversion * price)
# Parameters for different samples
num_samples = 5  # Number of samples
minimum_detectable_effect = 2.5  # MDE
prices = [100, 150, 150, 200, 250]  # Tariff prices
conversions = [0.15, 0.1, 0.1, 0.075, 0.06]  # Tariff conversions
significance_level = 0.05
power_level = 0.2
variances = [price ** 2 * conversion * (1 - conversion) for price, conversion in zip(prices, conversions)]

# Calculate minimum sample size for unequal variances
sample_size = min_sample_size(num_samples, minimum_detectable_effect, variances=variances,
                              significance_level=significance_level, power_level=power_level, equal_variance=False)
print(f'Sample size = {sample_size}')

# Testing samples with equal ARPU (Average Revenue Per User)
print('\nSamples with equal ARPU')
for _ in range(5):
    samples = [price * bernoulli.rvs(conversion, size=sample_size) for price, conversion in zip(prices, conversions)]
    hypothesis = test_on_marginal_distribution(samples, significance_level=significance_level)
    print(f'\tAccepted hypothesis H({hypothesis})')

# Testing where the last sample has higher ARPU by MDE
print('\nLast sample has higher ARPU by MDE')
for _ in range(5):
    samples = [price * bernoulli.rvs(conversion, size=sample_size) for price, conversion in
               zip(prices, conversions[:-1])]
    samples.append(
        prices[-1] * bernoulli.rvs(conversions[-1] + minimum_detectable_effect / prices[-1], size=sample_size))
    hypothesis = test_on_marginal_distribution(samples, significance_level=significance_level)
    print(f'\tAccepted hypothesis H({hypothesis})')
