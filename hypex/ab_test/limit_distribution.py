# Решаем задачу множественного тестирования
# Хотим выбрать лучшую выборку из k выборок по некоторой целевой метрике
# Пусть гипотеза H_i - iя выборка лучшая, i = 1, ..., k
# Пусть гипотеза H_0 - нет лучшей выборки
from typing import List, Union, Optional

from scipy.stats import norm


# Разработали метод, основанный на предельном распределении
# Он позволяет придерживаться заложенных вероятностей ошибок 1го и 2го рода
# Функции минимального размера выборки и критерия являются основными
# Функция квантиля предельного распределения случайной величины минимума используется в основных функциях


# Решаем задачу множественного тестирования
# Хотим выбрать лучшую выборку из k выборок по некоторой целевой метрике
# Пусть гипотеза H_i - iя выборка лучшая, i = 1, ..., k
# Пусть гипотеза H_0 - нет лучшей выборки

# Разработали метод, основанный на предельном распределении
# Он позволяет придерживаться заложенных вероятностей ошибок 1го и 2го рода

# Функции минимального размера выборки и критерия являются основными
# Функция квантиля предельного распределения случайной величины минимума используется в основных функциях


def quantile_of_marginal_distribution(num_samples: int, quantile_level: float, variances: List[float] = [1, 1, 1],
                                      equal_variance: bool = True) -> Union[float, List[float]]:
    """
    Calculate the quantile of the marginal distribution of the minimum.

    Parameters
    ----------
    num_samples : int
        Number of samples, an integer greater than 2.
    quantile_level : float
        Level of the quantile for the marginal distribution of the minimum.
    variances : list of float, optional
        List of variances of the samples, all of the same length. Number of samples should be greater than 2.
    equal_variance : bool, optional
        Indicates whether the variances are equal across samples.

    Returns
    -------
    float
        Quantile of the marginal distribution of the minimum if variances are equal.
    list of float
        List of quantiles of the marginal distribution of the minimum for each sample if variances are not equal.
    """
    iteration_size = 20000  # Количество итераций теста

    if equal_variance:
        reference_sample_index = 0  # в силу симметрии reference_sample_index по гипотезе H_0 возьмём reference_sample_index = 0 (первая выборка)
        t_values = []
        random_samples = norm.rvs(size=[iteration_size, num_samples], random_state=random_state)
        for sample in random_samples:
            min_t_value = np.inf
            for i in range(num_samples):
                if i != reference_sample_index:
                    t_value = (sample[reference_sample_index] - sample[i]) / np.sqrt(2)
                    min_t_value = min(min_t_value, t_value)
            t_values.append(min_t_value)
        return np.quantile(t_values, quantile_level)
    else:
        quantiles = []
        for j in range(num_samples):
            t_values = []
            random_samples = norm.rvs(size=[iteration_size, num_samples], random_state=random_state)
            for sample in random_samples:
                min_t_value = np.inf
                for i in range(num_samples):
                    if i != j:
                        t_value = sample[j] / np.sqrt(1 + variances[i] / variances[j]) - sample[i] / np.sqrt(
                            1 + variances[j] / variances[i])
                        min_t_value = min(min_t_value, t_value)
                t_values.append(min_t_value)
            quantiles.append(np.quantile(t_values, quantile_level))
        return quantiles


def test_on_marginal_distribution(samples: List[List[float]], significance_level: float = 0.05,
                                  equal_variance: bool = True,
                                  quantiles: Optional[Union[float, List[float]]] = None) -> int:
    num_samples = len(samples)
    sample_size = len(samples[0])

    samples_array = np.array(samples)
    means = np.mean(samples_array, axis=1)
    variances = np.var(samples_array, axis=1, ddof=1)

    if equal_variance:
        if quantiles is None:
            quantiles = quantile_of_marginal_distribution(num_samples, 1 - significance_level / num_samples)
        mean_diffs = means[:, None] - means
        t_values = np.sqrt(sample_size) * np.abs(mean_diffs) / np.sqrt(2 * variances[:, None])

        # Excluding diagonal elements for min calculation
        np.fill_diagonal(t_values, np.inf)
        max_t_value = np.max(np.min(t_values, axis=1))
        return np.argmax(max_t_value > quantiles) + 1 if max_t_value > quantiles else 0
    else:
        if quantiles is None:
            quantiles = quantile_of_marginal_distribution(num_samples, 1 - significance_level / num_samples, variances,
                                                          False)
        mean_diffs = means[:, None] - means
        sqrt_terms = np.sqrt(variances[:, None] + variances)
        t_values = np.sqrt(sample_size) * np.abs(mean_diffs) / sqrt_terms

        # Excluding diagonal elements for min calculation
        np.fill_diagonal(t_values, np.inf)
        max_t_values = np.max(np.min(t_values, axis=1), axis=1)
        return np.argmax(max_t_values > quantiles) + 1 if np.any(max_t_values > quantiles) else 0


def calculate_sample_variance(samples: List[List[float]]) -> List[float]:
    """Calculate variances for each sample."""
    samples_array = np.array(samples)
    return np.var(samples_array, axis=1, ddof=1)


def calculate_t_value(sample, i, j, variances, sample_size, mde):
    """Calculate t-value for a pair of samples."""
    return (sample[j] / np.sqrt(1 + variances[i] / variances[j]) -
            sample[i] / np.sqrt(1 + variances[j] / variances[i]) +
            mde * np.sqrt(sample_size / (variances[j] + variances[i])))


def calculate_power_for_sample(sample_index, variances, sample_size, mde, quantile_1, iteration_size=3000):
    """Calculate power for a single sample."""
    current_power = 0
    total_samples = norm.rvs(size=[iteration_size, len(variances)], random_state=42)
    for sample in total_samples:
        min_t_value = np.inf
        for i in range(len(variances)):
            if i != sample_index:
                t_value = calculate_t_value(sample, i, sample_index, variances, sample_size, mde)
                min_t_value = min(min_t_value, t_value)
        if min_t_value > quantile_1[sample_index]:
            current_power += 1
    return current_power / iteration_size


def min_sample_size(number_of_samples: int, minimum_detectable_effect: float, variances: Union[List[float], float],
                    significance_level: float = 0.05, power_level: float = 0.2, equal_variance: bool = True,
                    quantile_1: Optional[Union[float, List[float]]] = None, quantile_2: Optional[float] = None,
                    initial_estimate: Optional[int] = None) -> int:
    """
    Calculate the minimum sample size for statistical testing.

    Parameters
    ----------
    number_of_samples : int
        Number of samples, an integer greater than 2.
    minimum_detectable_effect : float
        Minimum detectable effect, a positive number.
    variances : Union[List[float], float]
        Estimated variances of the samples of equal length. List if variances are unequal, otherwise a single float.
    significance_level : float, optional
        Significance level, a number between 0 and 1.
    power_level : float, optional
        Power level (1 - beta), a number between 0 and 1.
    equal_variance : bool, optional
        Flag indicating whether variances are equal across samples.
    quantile_1 : Union[float, List[float]], optional
        Pre-computed quantile of the marginal distribution minimum if variances are equal, or a list of quantiles for each sample if not.
    quantile_2 : float, optional
        Pre-computed quantile for power level if variances are equal.
    initial_estimate : int, optional
        Initial guess for the sample size (for faster computation).

    Returns
    -------
    int
        The required sample size for each sample.
    """

    if equal_variance:
        if quantile_1 is None:
            quantile_1 = quantile_of_marginal_distribution(number_of_samples,
                                                           1 - significance_level / number_of_samples)
        if quantile_2 is None:
            quantile_2 = quantile_of_marginal_distribution(number_of_samples, power_level)
        return int(2 * variances * ((quantile_1 - quantile_2) / minimum_detectable_effect) ** 2) + 1
    else:
        iteration_size = 3000
        if quantile_1 is None:
            quantile_1 = quantile_of_marginal_distribution(number_of_samples,
                                                           1 - significance_level / number_of_samples, variances, False)
        sample_sizes = []
        for sample_index in range(number_of_samples):
            sample_size = initial_estimate or 0
            current_power = calculate_power_for_sample(sample_index, variances, sample_size, minimum_detectable_effect,
                                                       quantile_1, iteration_size)
            while current_power < 1 - power_level:
                sample_size += 100
                current_power = calculate_power_for_sample(sample_index, variances, sample_size,
                                                           minimum_detectable_effect, quantile_1, iteration_size)
            sample_sizes.append(sample_size)
        return max(sample_sizes)


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
