# Решаем задачу множественного тестирования
# Хотим выбрать лучшую выборку из k выборок по некоторой целевой метрике
# Пусть гипотеза H_i - iя выборка лучшая, i = 1, ..., k
# Пусть гипотеза H_0 - нет лучшей выборки

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


def quantile_of_marginal_distribution(num_samples, quantile_level, variances=None, iteration_size=20000):
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


def test_on_marginal_distribution(samples, significance_level=0.05, equal_variance=True, quantiles=None):
    num_samples = len(samples)  # Число выборок
    sample_size = len(samples[0])  # Размер выборки

    means = [np.mean(sample) for sample in samples]
    variances_q = [np.var(sample) * sample_size / (sample_size - 1) for sample in samples] if equal_variance else None
    variances = [np.var(sample) * sample_size / (sample_size - 1) for sample in samples]

    if quantiles is None:
        quantiles = quantile_of_marginal_distribution(num_samples=num_samples,
                                                      quantile_level=1 - significance_level / num_samples,
                                                      variances=variances_q)  # квантиль предельного распределения
    for j in range(num_samples):
        min_t_value = np.inf
        for i in range(num_samples):
            if i != j:
                t_value = np.sqrt(sample_size) * (means[j] - means[i]) / np.sqrt(variances[j] + variances[i])
                min_t_value = min(min_t_value, t_value)

        if min_t_value > quantiles[j]:
            return j + 1
    return 0


def min_sample_size(number_of_samples, minimum_detectable_effect, variances, significance_level=0.05, power_level=0.2,
                    equal_variance=True, quantile_1=None, quantile_2=None, initial_estimate=None):
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
