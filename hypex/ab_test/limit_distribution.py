from typing import List, Union, Optional

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

def calculate_comparisons(Z: np.ndarray, var: List[float], equal_variance: bool, k: int) -> List[float]:
    """
    Calculate the comparison values for a given sample array.

    Args:
        Z:
            The sample array.
        var:
            List of variances of the samples.
        equal_variance:
            Indicates if variances are equal.
        k:
            Number of samples.

    Returns:
        A list of comparison values.
    """
    if equal_variance:
        return [(Z[j] - Z[i]) / np.sqrt(2) for j in range(k) for i in range(k) if i != j]
    else:
        return [Z[j] / np.sqrt(1 + var[i] / var[j]) - Z[i] / np.sqrt(1 + var[j] / var[i]) for j in range(k) for i in
                range(k) if i != j]


def calculate_quantiles(k: int, var: List[float], equal_variance: bool, iter_size: int = 20000) -> List[float]:
    """
     Calculate quantiles for the distribution.

     Args:
         k:
             Number of samples.
         var:
             List of variances of the samples.
         equal_variance:
             Indicates if variances are equal.
         iter_size:
             Number of iterations for generating random variables.

     Returns:
         List of quantiles.
     """
    t_j = []
    total = norm.rvs(size=[iter_size, k])
    for Z in total:
        comparisons = calculate_comparisons(Z, var, equal_variance, k)
        t_j.append(min(comparisons))
    return t_j


def quantile_of_marginal_distribution(k: int, gamma: float, var: List[float] = None, equal_var: bool = True) -> Union[
    float, List[float]]:
    """
    Calculate the quantile of the marginal distribution of the minimum.

    Args:
        k:
            Number of samples, an integer greater than 2.
        gamma:
            Level of the quantile of the marginal distribution of the minimum.
        var:
            List of variances of the samples, all of the same length. Number of samples greater than 2.
            If None, defaults to a list of ones.
        equal_var:
            Indicates if variances are equal.

    Returns:
        If equal_var=True, returns the quantile of the marginal distribution of the minimum at level gamma.
        If equal_var=False, returns a set of quantiles of the marginal distribution of the minimum for each j at level gamma.
    """
    if var is None:
        var = [1] * k

    t_j = calculate_quantiles(k, var, equal_var)
    if equal_var:
        return np.quantile(t_j, gamma)
    else:
        return [np.quantile(t_j, gamma) for _ in range(k)]


def calculate_statistics(X: List[List[float]]) -> (List[float], List[float]):
    """Calculate mean and variance for each sample in X."""
    n = len(X[0])
    mean = [np.mean(sample) for sample in X]
    var = [np.var(sample) * n / (n - 1) for sample in X]
    return mean, var


def test_on_marginal_distribution(X: List[List[float]], alpha: float = 0.05, equal_var: bool = True,
                                  c: Union[float, List[float]] = None) -> int:
    """
    Function for testing based on the marginal distribution of the minimum.

    Args:
        X:
            List of samples of equal length. The number of samples is greater than 2.
        alpha:
            Significance level, a number between 0 and 1.
        equal_var:
            Indicates if variances are equal.
        c:
            Quantile of the marginal distribution of the minimum at level 1-alpha/len(X) if equal_var=True.
            Set of quantiles for each j at level 1-alpha/len(X) if equal_var=False.

    Returns:
        An integer from 0 to k - the number of the accepted hypothesis.
    """
    k = len(X)
    mean, var = calculate_statistics(X)

    if c is None:
        gamma = 1 - alpha / k
        c = quantile_of_marginal_distribution(k, gamma, var, equal_var)

    for j in range(k):
        t_j = np.inf
        for i in range(k):
            if i != j:
                t_ji = np.sqrt(len(X[0])) * (mean[j] - mean[i]) / np.sqrt(var[j] + var[i])
                t_j = min(t_j, t_ji)
        if (equal_var and t_j > c) or (not equal_var and t_j > c[j]):
            return j + 1

    return 0


def _calculate_sample_size_equal_var(k: int, d: float, var: float, alpha: float, beta: float, c_1: Optional[float],
                                     c_2: Optional[float]) -> int:
    """
    Calculate sample size for equal variances.
    (Docstring omitted for brevity)
    """
    if c_1 is None:
        c_1 = quantile_of_marginal_distribution(k, 1 - alpha / k, var, True)
    if c_2 is None:
        c_2 = quantile_of_marginal_distribution(k, beta, var, True)

    return int(2 * var * ((c_1 - c_2) / d) ** 2) + 1


def _calculate_sample_size_unequal_var(k: int, d: float, var: List[float], alpha: float, beta: float,
                                       c_1: Optional[List[float]], N: Optional[int]) -> int:
    """
    Calculate sample size for unequal variances.
    (Docstring omitted for brevity)
    """
    iter_size = 3000
    if c_1 is None:
        c_1 = quantile_of_marginal_distribution(k, 1 - alpha / k, var, False)

    N_ = []
    for j in range(k):
        n = N or 0
        power = 0
        while power < 1 - beta:
            n += 100
            power = sum(
                all(Z[j] / np.sqrt(1 + var[i] / var[j]) - Z[i] / np.sqrt(1 + var[j] / var[i]) + d * np.sqrt(
                    n / (var[j] + var[i])) > c_1[j]
                    for i in range(k) if i != j)
                for Z in norm.rvs(size=[iter_size, k])
            ) / iter_size
        N_.append(n)
    return max(N_)


def min_sample_size(k, d, var, alpha=0.05, beta=0.2, equal_var=True, c_1=None, c_2=None, N=None):
    """Функция для подсчёта минимального размера выборки

    Parameters
    ----------
    k : int
        Количество выборок, целое число больше 2
    d : float
        Minimum Detectable Effect, положительное число
    var : list of float, если equal_var=False
          float, если equal_var=True
        Оценка дисперсии выборок одинаковой длины при H(0). Количество выборок больше 2
    alpha : float, optional
        Уровень значимости, число от 0 до 1
    beta : float, optional
        1 - мощность, число от 0 до 1
    equal_var : bool, optional
        Равенство дисперсий
    c_1 : optional
            float, если equal_var=True
                Квантиль предельного распределения минимума уровня 1-alpha/len(X)
            list of float, если equal_var=False
                Набор квантилей предельного распределения минимума для каждого j уровня 1-alpha/len(X)
    c_2 : optional
            float, если equal_var=True
                Квантиль предельного распределения минимума уровня beta
            None, если equal_var=False
    N : int, optional
        Нижняя граница для размера (для более быстрой работы программы)

    # В нашем случае все выборки будут одного размера

    Returns
    -------
    int
        Число n - размер одной выборки
    """
    if equal_var:
        return _calculate_sample_size_equal_var(k, d, var, alpha, beta, c_1, c_2)
    else:
        return _calculate_sample_size_unequal_var(k, d, var, alpha, beta, c_1, N)


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
n = min_sample_size(k=k, d=d, var=p * (1 - p), alpha=alpha, beta=beta, equal_var=True)
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
n = min_sample_size(k=k, d=d, var=var, alpha=alpha, beta=beta, equal_var=False)
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
