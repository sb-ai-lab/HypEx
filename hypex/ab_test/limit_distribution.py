from typing import List, Union, Optional, Tuple

import numpy as np
from scipy.stats import norm


# Решаем задачу множественного тестирования
# Хотим выбрать лучшую выборку из k выборок по некоторой целевой метрике
# Пусть гипотеза H_i - iя выборка лучшая, i = 1, ..., k
# Пусть гипотеза H_0 - нет лучшей выборки

# Разработали метод, основанный на предельном распределении
# Он позволяет придерживаться заложенных вероятностей ошибок 1го и 2го рода

# Функции минимального размера выборки и критерия являются основными
# Функция квантиля предельного распределения случайной величины минимума используется в основных функциях


def calculate_comparisons(sample_array: np.ndarray, variances: List[float], equal_variance: bool) -> np.ndarray:
    """
    Calculate the comparison values for a given sample array.

    Args:
        sample_array:
            The sample array representing a specific observation across different samples.
        variances:
            List of variances of the samples.
        equal_variance:
            Indicates if variances are equal.

    Returns:
        A NumPy array of comparison values.
    """
    num_samples = len(sample_array)
    if equal_variance:
        diff_matrix = sample_array[:, np.newaxis] - sample_array
        comparison_values = diff_matrix / np.sqrt(2)
    else:
        variances = np.array(variances)
        scaling_factors = 1 / np.sqrt(1 + variances[:, np.newaxis] / variances)
        sample_array_2d = sample_array[:, np.newaxis]
        scaled_sample_array = sample_array_2d * scaling_factors
        comparison_values = scaled_sample_array - scaled_sample_array.T

    # Remove diagonal elements (i != j)
    np.fill_diagonal(comparison_values, np.nan)
    return comparison_values[~np.isnan(comparison_values)]


def calculate_min_comparisons(sample_matrix: np.ndarray, variances: List[float], equal_variance: bool) -> np.ndarray:
    """
    Calculate the minimum comparison values for each row in a sample matrix.

    Args:
        sample_matrix:
            A matrix of random samples, each row representing a set of observations across samples.
        variances:
            List of variances of the samples.
        equal_variance:
            Indicates if variances are equal.

    Returns:
        A NumPy array of minimum comparison values for each row.
    """
    min_comparisons = np.array([min(calculate_comparisons(row, variances, equal_variance)) for row in sample_matrix])
    return min_comparisons


def calculate_quantiles(num_samples: int, variances: List[float], equal_variance: bool, iterations: int = 20000) -> \
        List[float]:
    """
    Calculate quantiles for the distribution.

    Args:
        num_samples:
            Number of samples.
        variances:
            List of variances of the samples.
        equal_variance:
            Indicates if variances are equal.
        iterations:
            Number of iterations for generating random variables.

    Returns:
        List of quantiles.
    """
    random_samples = norm.rvs(size=[iterations, num_samples])
    min_comparison_values = calculate_min_comparisons(random_samples, variances, equal_variance)
    return list(min_comparison_values)


def quantile_of_marginal_distribution(num_samples: int, gamma_level: float,
                                      variances: Union[List[float], np.ndarray] = None,
                                      equal_variance: bool = True) -> Union[float, List[float]]:
    """
    Calculate the quantile of the marginal distribution of the minimum.

    Args:
        num_samples:
            Number of samples, an integer greater than 2.
        gamma_level:
            Level of the quantile of the marginal distribution of the minimum.
        variances:
            List of variances of the samples, all of the same length. Number of samples greater than 2.
            If None, defaults to a list of ones.
        equal_variance:
            Indicates if variances are equal.

    Returns:
        If equal_variance=True, returns the quantile of the marginal distribution of the minimum at level gamma_level.
        If equal_variance=False, returns a set of quantiles of the marginal distribution of the minimum for each sample at level gamma_level.
    """
    if variances is None:
        variances = np.ones(num_samples)

    min_comparison_values = calculate_quantiles(num_samples, variances, equal_variance)
    if equal_variance:
        return np.quantile(min_comparison_values, gamma_level)
    else:
        return np.quantile(min_comparison_values, [gamma_level] * num_samples)


def calculate_statistics(samples: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and variance for each sample in a collection of samples.

    Args:
        samples: A list of lists, where each inner list represents a sample.

    Returns:
        A tuple containing two arrays:
        - The first array is the means of each sample.
        - The second array is the variances of each sample.
    """
    samples_array = np.array([np.array(sample) for sample in samples])
    means = np.mean(samples_array, axis=1)
    variances = np.var(samples_array, axis=1, ddof=1)  # ddof=1 for sample variance
    return means, variances


def test_on_marginal_distribution(samples: List[List[float]], alpha: float = 0.05, equal_var: bool = True,
                                  quantiles: Union[float, List[float]] = None) -> int:
    """
    Function for testing based on the marginal distribution of the minimum.

    Args:
        samples:
            List of samples of equal length. The number of samples is greater than 2.
        alpha:
            Significance level, a number between 0 and 1.
        equal_var:
            Indicates if variances are equal.
        quantiles:
            Quantile of the marginal distribution of the minimum at level 1-alpha/len(samples) if equal_var=True.
            Set of quantiles for each j at level 1-alpha/len(samples) if equal_var=False.

    Returns:
        An integer from 0 to the number of samples - the number of the accepted hypothesis.
    """
    num_samples = len(samples)
    means, variances = calculate_statistics(samples)
    sample_size = len(samples[0])

    if quantiles is None:
        gamma_level = 1 - alpha / num_samples
        quantiles = quantile_of_marginal_distribution(num_samples, gamma_level, variances, equal_var)

    for j in range(num_samples):
        comparison_values = np.sqrt(sample_size) * (means[j] - means) / np.sqrt(variances[j] + variances)
        comparison_values[j] = np.inf  # Ignore self-comparison
        t_j = np.min(comparison_values)

        if (equal_var and t_j > quantiles) or (not equal_var and t_j > quantiles[j]):
            return j + 1

    return 0


def _calculate_sample_size_equal_var(num_samples: int, minimum_effect: float, variance: float, alpha: float,
                                     beta: float, quantile_1: Optional[float], quantile_2: Optional[float]) -> int:
    """
    Calculate the required sample size assuming equal variances across samples.

    Args:
        num_samples: The number of samples.
        minimum_effect: The minimum detectable effect size (d).
        variance: The common variance of the samples.
        alpha: The significance level.
        beta: The probability of Type II error (1 - power of the test).
        quantile_1: The quantile of the marginal distribution for 1 - alpha/k. If None, it will be calculated.
        quantile_2: The quantile of the marginal distribution for beta. If None, it will be calculated.

    Returns:
        The calculated sample size.
    """
    if quantile_1 is None:
        quantile_1 = quantile_of_marginal_distribution(num_samples, 1 - alpha / num_samples, variance, True)
    if quantile_2 is None:
        quantile_2 = quantile_of_marginal_distribution(num_samples, beta, variance, True)

    return int(2 * variance * ((quantile_1 - quantile_2) / minimum_effect) ** 2) + 1


def _calculate_sample_size_unequal_var(num_samples: int, minimum_effect: float, variances: List[float], alpha: float,
                                       beta: float, quantile_thresholds: Optional[List[float]],
                                       initial_estimate: Optional[int], increment: int = 100,
                                       iterations: int = 3000) -> int:
    """
    Calculate the required sample size for unequal variances across samples.

    Args:
        num_samples: The number of samples.
        minimum_effect: The minimum detectable effect size.
        variances: A list of variances for each sample.
        alpha: The significance level.
        beta: The probability of Type II error (1 - power of the test).
        quantile_thresholds: Precomputed quantiles for the marginal distribution. If None, they will be calculated.
        initial_estimate: An initial estimate of sample size. If None, starts from 0.
        increment: The step size to increment the sample size in each iteration.
        iterations: The number of iterations for the simulation.

    Returns:
        The calculated sample size.
    """
    if quantile_thresholds is None:
        quantile_thresholds = quantile_of_marginal_distribution(num_samples, 1 - alpha / num_samples, variances, False)

    sample_sizes = []
    for j in range(num_samples):
        sample_size = initial_estimate or 0
        while True:
            sample_size += increment
            random_samples = norm.rvs(size=[iterations, num_samples])
            power = np.mean([
                np.all([
                    (random_samples[i, j] / np.sqrt(1 + variances[i] / variances[j]) - random_samples[i, i] / np.sqrt(
                        1 + variances[j] / variances[i]) + minimum_effect * np.sqrt(
                        sample_size / (variances[j] + variances[i]))) > quantile_thresholds[j]
                    for i in range(num_samples) if i != j
                ])
                for i in range(iterations)
            ])
            if power >= 1 - beta:
                break
        sample_sizes.append(sample_size)
    return max(sample_sizes)


def min_sample_size(num_samples: int, minimum_effect: float, variances: Union[List[float], float], alpha: float = 0.05,
                    beta: float = 0.2, equal_variance: bool = True,
                    quantile_1: Optional[Union[float, List[float]]] = None, quantile_2: Optional[float] = None,
                    initial_estimate: Optional[int] = None) -> int:
    """
    Calculate the minimum sample size required for a statistical test.

    Args:
        num_samples: Number of samples, an integer greater than 2.
        minimum_effect: The Minimum Detectable Effect (MDE), a positive number.
        variances: Estimates of the variances of the samples. It's a list of floats if variances are unequal, or a single float if variances are equal.
        alpha: Significance level, a number between 0 and 1.
        beta: 1 minus the power of the test, a number between 0 and 1.
        equal_variance: Indicates if variances are equal.
        quantile_1: Quantile of the marginal distribution at level 1-alpha/num_samples. For equal variances, it's a float; for unequal variances, it's a list of floats.
        quantile_2: Quantile of the marginal distribution at level beta. It's a float for equal variances and None for unequal variances.
        initial_estimate: An optional lower bound for the sample size to speed up the calculation.

    Returns:
        The calculated minimum sample size.
    """
    if equal_variance:
        return _calculate_sample_size_equal_var(num_samples, minimum_effect, variances, alpha, beta, quantile_1,
                                                quantile_2)
    else:
        return _calculate_sample_size_unequal_var(num_samples, minimum_effect, variances, alpha, beta, quantile_1,
                                                  initial_estimate)


# Применение метода

# Пример 1
# Рассмотрим множественный тест для выявления выборки с лучшей конверсии (распределение Бернулли)
# В этом случае при гипотезе H_0, равенстве конверсий на всех выборках, дисперсии равны, 
# поскольку дисперсия зависит только от конверсии

from scipy.stats import bernoulli

k = 10  # число выборок
d = 0.05  # MDE
p = 0.3  # предполагаемая конверсия (параметр распределения Бернулли)
alpha = 0.05  # уровень значимости
beta = 0.2  # 1 - мощность

# считаем минимальный размер выборки
n = min_sample_size(k, d, p * (1 - p), alpha=alpha, beta=beta, equal_variance=True)
print(f'Размер выборки = {n}')

# Попробуем сгенерировать выборки и посмотреть результат тестирования
N = 5
# все выборки имеют одинаковую конверсию
print('\nВсе выборки имеют одинаковую конверсию')
for _ in range(N):
    X = bernoulli.rvs(p, size=[k, n])
    hyp = test_on_marginal_distribution(X, alpha=alpha)
    print(f'\tПринята гипотеза H({hyp})')

# десятая выборка имеет большую на MDE конверсию
print('\nДесятая выборка имеет большую на MDE конверсию')
for _ in range(N):
    X = []
    for i in range(k - 1):
        X += [bernoulli.rvs(p, size=n)]
    X += [bernoulli.rvs(p + d, size=n)]
    hyp = test_on_marginal_distribution(X, alpha=alpha)
    print(f'\tПринята гипотеза H({hyp})')

# Пример 2
# Рассмотрим множественный тест для выявления выборки с лучшим доходом на клиента (конверсия * цена)
# В этом случае при гипотезе H_0, равенстве ARPU на всех выборках, дисперсии не равны, 
# поскольку при гипотезе H_0 при разных ценах получаем разные конверсии

k = 5  # число выборок
d = 2.5  # MDE
# в среднем ARPU = 15 рублей
Price = [100, 150, 150, 200, 250]  # цены тарифов
conv = [0.15, 0.1, 0.1, 0.075, 0.06]  # конверсии тарифов
alpha = 0.05  # уровень значимости
beta = 0.2  # 1 - мощность
var = []  # дисперсии тарифов
for i in range(k):
    var += [Price[i] ** 2 * conv[i] * (1 - conv[i])]

# считаем минимальный размер выборки
n = min_sample_size(k, d, var, alpha=alpha, beta=beta, equal_variance=False)
print(f'Размер выборки = {n}')

# Попробуем сгенерировать выборки и посмотреть результат тестирования
N = 5
# все выборки имеют одинаковый ARPU
print('\nВсе выборки имеют одинаковый ARPU')
for _ in range(N):
    X = []
    for i in range(k):
        X += [Price[i] * bernoulli.rvs(conv[i], size=n)]
    hyp = test_on_marginal_distribution(X, alpha=alpha)
    print(f'\tПринята гипотеза H({hyp})')

# десятая выборка имеет больший на MDE ARPU
print('\nДесятая выборка имеет больший на MDE ARPU')
for _ in range(N):
    X = []
    for i in range(k - 1):
        X += [Price[i] * bernoulli.rvs(conv[i], size=n)]
    X += [Price[k - 1] * bernoulli.rvs(conv[k - 1] + d / Price[k - 1], size=n)]
    hyp = test_on_marginal_distribution(X, alpha=alpha)
    print(f'\tПринята гипотеза H({hyp})')
