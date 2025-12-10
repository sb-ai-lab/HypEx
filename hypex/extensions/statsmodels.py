from __future__ import annotations

import numpy as np
from scipy.stats import norm  # type: ignore
from statsmodels.stats.multitest import multipletests  # type: ignore

from ..dataset import Dataset, DatasetAdapter, StatisticRole
from ..utils import ID_SPLIT_SYMBOL, ABNTestMethodsEnum
from .abstract import Extension


class MultiTest(Extension):
    def __init__(self, method: ABNTestMethodsEnum, alpha: float = 0.05):
        self.method = method
        self.alpha = alpha
        super().__init__()

    def _calc_pandas(self, data: Dataset, **kwargs):
        p_values = data.data.values.flatten()
        new_pvalues = multipletests(
            p_values, method=self.method.value, alpha=self.alpha, **kwargs
        )
        return DatasetAdapter.to_dataset(
            {
                "field": [i.split(ID_SPLIT_SYMBOL)[2] for i in data.index],
                "test": [i.split(ID_SPLIT_SYMBOL)[0] for i in data.index],
                "old p-value": p_values,
                "new p-value": new_pvalues[1],
                "correction": [
                    j / i if j != 0 else 0.0 for i, j in zip(new_pvalues[1], p_values)
                ],
                "rejected": new_pvalues[0],
            },
            StatisticRole(),
        )


class MultitestQuantile(Extension):
    def __init__(
        self,
        alpha: float = 0.05,
        iteration_size: int = 20000,
        equal_variance: bool = True,
        random_state: int | None = None,
    ):
        self.alpha = alpha
        self.iteration_size = iteration_size
        self.equal_variance = equal_variance
        self.random_state = random_state
        super().__init__()

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
                return DatasetAdapter.to_dataset(
                    {"field": target_field, "accepted hypothesis": j + 1},
                    StatisticRole(),
                )
        return DatasetAdapter.to_dataset(
            {"field": target_field, "accepted hypothesis": 0}, StatisticRole()
        )

    def quantile_of_marginal_distribution(
        self,
        num_samples: int,
        quantile_level: float,
        variances: list[float] | None = None,
    ) -> list[float]:
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
                            if variances is None:
                                raise ValueError("variances is needed for execution")
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
    num_samples: int,
    mde: float,
    variances: list[float] | float,
    power: float = 0.2,
    quantile_1: float | list[float] | None = None,
    quantile_2: float | list[float] | None = None,
    initial_estimate: int = 0,
    power_iteration_size: int = 3000,
    alpha: float = 0.05,
    iteration_size: int = 5000,
    equal_variance: bool = True,
    random_state: int | None = 42,
):
    """Estimate the minimum sample size for a multi-armed test with multiplicity control.

    This function computes the minimal per-group sample size required to detect a given
    minimal detectable effect (MDE) in a multi-group experiment while controlling the
    family-wise error rate (FWER). It relies on the marginal distribution of the
    minimum t-statistic across groups, estimated via Monte Carlo simulation using
    :class:`MultitestQuantile` and its ``quantile_of_marginal_distribution`` method.

    When ``equal_variance=True``, an analytic approximation is used based on the
    variance and critical quantiles. When ``equal_variance=False``, a simulation-based
    search is performed for each group to achieve the desired power, and the maximum
    required sample size across all groups is returned.

    Args:
        num_samples (int): Number of groups (arms) in the experiment.
        mde (float): Minimal detectable effect (difference in means) to be detected
            between any pair of groups.
        variances (Union[List[float], float]): Variance(s) of the target metric.
            If ``equal_variance=True``, this can be a single float (common variance) or
            a list of length ``num_samples``. If ``equal_variance=False``, this must be
            a list of length ``num_samples``, providing a separate variance for each group.
        power (float, optional): Type II error rate :math:`\\beta`. The target power is
            ``1 - power``. For example, ``power=0.2`` corresponds to 80% power.
            Defaults to 0.2.
        quantile_1 (Union[float, List[float], None], optional): Precomputed critical
            quantile(s) of the marginal distribution of the minimum t-statistic under
            the null hypothesis, typically at level ``1 - alpha / num_samples``.
            If a float is provided, it is broadcast to all groups. If ``None``,
            it is estimated internally via Monte Carlo. Defaults to None.
        quantile_2 (Union[float, List[float], None], optional): Precomputed quantile(s)
            of the marginal distribution of the minimum t-statistic under the alternative
            hypothesis, typically at level ``power``. If a float is provided, it is
            broadcast to all groups. If ``None``, it is estimated internally via
            Monte Carlo. Defaults to None.
        initial_estimate (int, optional): Initial sample size estimate used as a starting
            point in the simulation-based search when ``equal_variance=False``.
            Defaults to 0.
        power_iteration_size (int, optional): Number of Monte Carlo iterations used in
            the inner power estimation loop when ``equal_variance=False``.
            Larger values yield more stable estimates but increase computation time.
            Defaults to 3000.
        alpha (float, optional): Significance level for controlling the family-wise
            error rate. Used to compute the critical quantile under the null as
            ``1 - alpha / num_samples``. Defaults to 0.05.
        iteration_size (int, optional): Number of Monte Carlo samples used inside
            :class:`MultitestQuantile` to estimate marginal quantiles of the minimum
            t-statistic (both under null and alternative). Defaults to 5000.
        equal_variance (bool, optional): Whether to assume equal variances across
            all groups. If ``True``, an analytic formula is used and ``variances`` may
            be passed as a single float. If ``False``, group-specific variances are used
            and a simulation-based search is performed. Defaults to True.
        random_state (Optional[int], optional): Random seed for reproducible Monte Carlo
            simulations. If ``None``, randomness is not seeded. Defaults to 42.

    Returns:
        Union[int, Dict[str, int]]:
            - If ``equal_variance=True``: an integer representing the minimal required
              sample size per group.
            - If ``equal_variance=False``: a dictionary with a single key
              ``"min sample size"`` containing the minimal required sample size
              across all groups.

    Examples
    --------
    .. code-block:: python

        # Minimum sample size for 3 groups with equal variances (80% power, alpha=0.05)
        n = min_sample_size(
            num_samples=3,
            mde=0.1,
            variances=1.5,
            power=0.2,          # 80% power
            alpha=0.05,
            equal_variance=True,
        )

        # Minimum sample size for 3 groups with different variances
        result = min_sample_size(
            num_samples=3,
            mde=0.1,
            variances=[1.2, 1.5, 2.0],
            power=0.2,              # 80% power
            alpha=0.05,
            equal_variance=False,
            initial_estimate=1000,
            power_iteration_size=5000,
            iteration_size=5000,
            random_state=42,
        )
        min_n = result["min sample size"]
    """

    multitest = MultitestQuantile(
        alpha=alpha,
        iteration_size=iteration_size,
        equal_variance=equal_variance,
        random_state=random_state,
    )
    if not isinstance(variances, list) and not equal_variance:
        raise TypeError("variances must be a list when equal_variance is False")
    
    if isinstance(quantile_1, float):
        quantile_1 = np.full(num_samples, quantile_1).tolist()
    if isinstance(quantile_2, float):
        quantile_2 = np.full(num_samples, quantile_2).tolist()

    quantile_1 = quantile_1 or multitest.quantile_of_marginal_distribution(
        num_samples=num_samples,
        quantile_level=1 - multitest.alpha / num_samples,
        variances=variances if isinstance(variances, list) else [variances],
    )
    quantile_2 = quantile_2 or multitest.quantile_of_marginal_distribution(
        num_samples=num_samples,
        quantile_level=power,
    )
    
    if multitest.equal_variance:
        var = variances[0] if isinstance(variances, list) else variances
        return int(2 * var * ((quantile_1[0] - quantile_2[0]) / mde) ** 2) + 1

    sizes: list[int] = []
    for index in range(num_samples):
        size = initial_estimate
        current_power = 0.0

        while current_power < 1 - power:
            size += 100
            current_power = 0

            total_samples = norm.rvs(
                size=[power_iteration_size, num_samples],
                random_state=multitest.random_state,
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
                            + mde * np.sqrt(size / (variances[index] + variances[i]))
                        )
                        min_t_value = min(min_t_value, t_value)

                if min_t_value > quantile_1[index]:
                    current_power += 1

            current_power /= power_iteration_size

        sizes.append(size)

    return {"min sample size": int(np.max(sizes))}