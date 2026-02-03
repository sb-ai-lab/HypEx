from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm

from ..dataset import ABCRole, Dataset, ExperimentData, TargetRole, TreatmentRole
from ..extensions import MultitestQuantile
from ..utils import NotSuitableFieldError
from ..utils.adapter import Adapter
from .executor import Calculator


class MinSampleSize(Calculator):
    """A calculator for estimating the minimum required sample size for multi-group comparisons.

    This class estimates the minimum per-group sample size needed to achieve a desired statistical
    power for detecting a specified minimum detectable effect (MDE) when comparing multiple groups
    (e.g., control vs one or more test groups). Quantiles used in the calculation can be provided
    explicitly or estimated internally using `MultitestQuantile`.

    The calculator supports both:
      - **Equal-variance mode** (`equal_variance=True`): closed-form sample size approximation based on
        a pooled/assumed common variance.
      - **Unequal-variance mode** (`equal_variance=False`): simulation-based sample size search that
        accounts for different variances across groups.

    Args:
        grouping_role (ABCRole | None, optional): Role used to locate the grouping (treatment) field
            in the dataset. If not provided, defaults to `TreatmentRole()`.
        key (Any, optional): Key used by the base `Calculator` for storing results. Defaults to "".
        mde (float): Minimum Detectable Effect (absolute effect size in the same units as the target
            metric) to be detected.
        power (float, optional): Power-related quantile level used in the internal quantile computation
            (kept consistent with the original function implementation). Defaults to 0.2.
        quantile_1 (float | list[float] | None, optional): Precomputed critical quantile(s) for the
            multiple testing threshold. If a float is provided, it is broadcast to all groups. If
            None, computed via `MultitestQuantile.quantile_of_marginal_distribution`. Defaults to None.
        quantile_2 (float | list[float] | None, optional): Precomputed quantile(s) used for power
            calibration. If a float is provided, it is broadcast to all groups. If None, computed
            via `MultitestQuantile.quantile_of_marginal_distribution`. Defaults to None.
        initial_estimate (int, optional): Starting sample size guess (used only when `equal_variance=False`).
            Defaults to 0.
        power_iteration_size (int, optional): Number of Monte Carlo iterations used to estimate achieved
            power during the sample size search (`equal_variance=False`). Defaults to 3000.
        alpha (float, optional): Significance level used in multiple testing quantile computation.
            Defaults to 0.05.
        iteration_size (int, optional): Internal iteration size for `MultitestQuantile` quantile estimation.
            Defaults to 5000.
        equal_variance (bool, optional): If True, uses the equal-variance closed-form approximation.
            If False, uses the unequal-variance simulation-based search. Defaults to False.
        random_state (int | None, optional): Random seed for reproducibility of Monte Carlo simulation
            and quantile estimation. Defaults to 42.
        variances (list[float] | float | None, optional): Variance specification. If provided:
            - when `equal_variance=True`, may be a single float (common variance) or a list (first/pooled
              usage depends on implementation).
            - when `equal_variance=False`, must be a list of variances per group (order matching grouping).
            If None, variances are estimated from the grouped data for each target metric. Defaults to None.

    Examples
    --------
    .. code-block:: python

        ds = Dataset(
            data="data.csv",
            roles={
                "user_id": InfoRole(int),
                "treat": TreatmentRole(),
                "pre_spends": TargetRole(),
                "post_spends": TargetRole(),
            },
        )

        mss = MinSampleSize(mde=10.0, alpha=0.05, equal_variance=True)
        result = mss.calc(data=ds)
    """

    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        key: Any = "",
        *,
        mde: float,
        power: float = 0.2,
        quantile_1: float | list[float] | None = None,
        quantile_2: float | list[float] | None = None,
        initial_estimate: int = 0,
        power_iteration_size: int = 3000,
        alpha: float = 0.05,
        iteration_size: int = 5000,
        equal_variance: bool = False,
        random_state: int | None = 42,
        variances: list[float] | float | None = None,
    ):
        super().__init__(key=key)
        self.grouping_role = grouping_role or TreatmentRole()

        self.mde = mde
        self.power = power
        self.quantile_1 = quantile_1
        self.quantile_2 = quantile_2
        self.initial_estimate = initial_estimate
        self.power_iteration_size = power_iteration_size
        self.alpha = alpha
        self.iteration_size = iteration_size
        self.equal_variance = equal_variance
        self.random_state = random_state
        self.variances = variances

    @property
    def search_types(self) -> list[type] | None:
        return [int, float]

    def _get_fields(self, data: Dataset) -> tuple[list[str], list[str]]:
        group_field = data.search_columns(self.grouping_role, search_types=None)
        target_fields = data.search_columns(
            [TargetRole()], search_types=self.search_types
        )
        return group_field, target_fields

    @staticmethod
    def _variance_by_group(
        grouping_data: list[tuple[str, Dataset]],
        target_field: str,
    ) -> list[float]:
        vars_: list[float] = []
        for _, ds in grouping_data:
            vars_.append(float(ds[target_field].var()))
        return vars_

    @classmethod
    def _inner_function(
        cls,
        *,
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
    ) -> int:
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
        assert isinstance(variances, list)

        for index in range(num_samples):
            size = initial_estimate
            current_power = 0.0

            while current_power < 1 - power:
                size += 100
                current_power = 0.0

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
                                + mde
                                * np.sqrt(size / (variances[index] + variances[i]))
                            )
                            min_t_value = min(min_t_value, t_value)

                    if min_t_value > quantile_1[index]:
                        current_power += 1.0

                current_power /= float(power_iteration_size)

            sizes.append(size)

        return int(np.max(sizes))

    def calc(self, data: Dataset) -> dict:
        group_field, target_fields = self._get_fields(data=data)

        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )

        if not target_fields and data.tmp_roles:
            raise Exception("No target fields in data")

        gf = Adapter.to_list(group_field)
        grouping_data = list(data.groupby(gf))

        if len(grouping_data) <= 1:
            raise NotSuitableFieldError(gf, "Grouping")

        result: dict = {}
        sizes: list[int] = []

        for field in target_fields:
            if self.variances is None:
                group_vars = self._variance_by_group(grouping_data, target_field=field)
                variances_used: list[float] | float = (
                    float(np.mean(group_vars)) if self.equal_variance else group_vars
                )
            else:
                variances_used = self.variances

            n = self._inner_function(
                num_samples=len(grouping_data),
                mde=self.mde,
                variances=variances_used,
                power=self.power,
                quantile_1=self.quantile_1,
                quantile_2=self.quantile_2,
                initial_estimate=self.initial_estimate,
                power_iteration_size=self.power_iteration_size,
                alpha=self.alpha,
                iteration_size=self.iteration_size,
                equal_variance=self.equal_variance,
                random_state=self.random_state,
            )

            result[field] = {"min sample size": n}
            sizes.append(n)

        result["overall"] = (
            {"min sample size": int(max(sizes))} if sizes else {"min sample size": 0}
        )

        return result

    def execute(self, data: ExperimentData) -> dict:
        return self.calc(data.ds)
