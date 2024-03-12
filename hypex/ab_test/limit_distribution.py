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


def quantile_of_marginal_distribution(
    num_samples: int,
    quantile_level: float,
    variances: Optional[Union[List, float]] = None,
    iteration_size: int = 20000,
    random_state: int = 42,
):
    """Calculate the quantile(s) of the marginal distribution for minimum t-values across multiple comparisons.

    This function generates random samples from a normal distribution and computes t-values for comparisons either
    with equal variances (if variances are not provided) or with specified unequal variances (if variances are provided).
    It then determines the quantile of interest for the distribution of the minimum t-values from these comparisons.

    Args:
        num_samples: The number of samples/groups to compare.
        quantile_level: The quantile level to compute for the marginal distribution (e.g., 0.95 for the 95th percentile).
        variances: A list of variances for each sample/group. If None, equal variances are assumed.
        iteration_size: The number of iterations/random samples to generate for the simulation.
        random_state: Optional parameter for data randomisation.

    Returns:
       The quantile of interest for the marginal distribution of the minimum t-values. Returns
       a single float if variances are assumed equal (or not provided) or a list of floats
       with quantiles for each sample if variances are provided and unequal.
    """
    np.random.seed(random_state)
    total = norm.rvs(size=[iteration_size, num_samples])

    if variances is None:
        j = 0
        t_values = [
            min(
                [
                    (total[l][j] - total[l][i]) / np.sqrt(2)
                    for i in range(num_samples)
                    if i != j
                ]
            )
            for l in range(iteration_size)
        ]
        return np.quantile(t_values, quantile_level)

    quantiles = []
    for j in range(num_samples):
        t_values = [
            min(
                [
                    total[l][j] / np.sqrt(1 + variances[i] / variances[j])
                    - total[l][i] / np.sqrt(1 + variances[j] / variances[i])
                    for i in range(num_samples)
                    if i != j
                ]
            )
            for l in range(iteration_size)
        ]
        quantiles += [np.quantile(t_values, quantile_level)]
    return quantiles


def test_on_marginal_distribution(
    samples: List[np.ndarray],
    significance_level: float = 0.05,
    equal_variance: bool = True,
    quantiles: Optional[Union[float, List[float]]] = None,
) -> int:
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

    num_samples, sample_size = len(samples), len(samples[0])

    means = [np.mean(sample) for sample in samples]
    variances = [np.var(sample) * sample_size / (sample_size - 1) for sample in samples]
    variances_q = variances if equal_variance else None

    if quantiles is None or isinstance(quantiles, float):
        quantiles = quantile_of_marginal_distribution(
            num_samples=num_samples,
            quantile_level=1 - significance_level / num_samples,
            variances=variances_q,
        )
    for j in range(num_samples):
        total_j = np.inf
        for i in range(num_samples):
            if i != j:
                total = (
                    np.sqrt(sample_size)
                    * (means[j] - means[i])
                    / np.sqrt(variances[j] + variances[i])
                )
                total_j = total if total < total_j else total_j
        if total_j > quantiles[j]:
            return j + 1
    return 0


def min_sample_size(
    number_of_samples: int,
    minimum_detectable_effect: float,
    variances: Union[List[float], float],
    significance_level: float = 0.05,
    power_level: float = 0.2,
    equal_variance: bool = True,
    quantile_1: Optional[Union[float, List[float]]] = None,
    quantile_2: Optional[float] = None,
    initial_estimate: Optional[int] = None,
    random_state: int = 42,
) -> int:
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
        random_state: Optional parameter for data randomisation.

    Returns:
        The minimum sample size required per sample/group.
    """
    if equal_variance:

        if quantile_1 is None:
            quantile_1 = quantile_of_marginal_distribution(
                num_samples=number_of_samples,
                quantile_level=1 - significance_level / number_of_samples,
            )  # quantile of the marginal distribution 1-alpha/k

        if quantile_2 is None:
            quantile_2 = quantile_of_marginal_distribution(
                num_samples=number_of_samples, quantile_level=power_level
            )  # quantile of the marginal distribution beta

        return (
            int(
                2
                * variances
                * ((quantile_1 - quantile_2) / minimum_detectable_effect) ** 2
            )
            + 1
        )
    else:
        iteration_size = 3000  # number of iterations
        if quantile_1 is None:
            quantile_1 = quantile_of_marginal_distribution(
                num_samples=number_of_samples,
                quantile_level=1 - significance_level / number_of_samples,
                variances=variances,
            )  # set of quantiles of the marginal distribution
        sample_sizes = []  # for sample sizes
        for j in range(number_of_samples):
            sample_size = initial_estimate or 0
            current_power = 0  # power
            while current_power < 1 - power_level:
                sample_size += 100
                current_power = 0
                total_samples = norm.rvs(
                    size=[iteration_size, number_of_samples], random_state=random_state
                )
                for sample in total_samples:
                    min_t_value = np.inf
                    for i in range(number_of_samples):
                        if i != j:
                            t_value = (
                                sample[j] / np.sqrt(1 + variances[i] / variances[j])
                                - sample[i] / np.sqrt(1 + variances[j] / variances[i])
                                + minimum_detectable_effect
                                * np.sqrt(sample_size / (variances[j] + variances[i]))
                            )
                            min_t_value = min(min_t_value, t_value)
                    if min_t_value > quantile_1[j]:
                        current_power += 1
                current_power /= iteration_size
            sample_sizes.append(sample_size)
        return np.max(sample_sizes)
