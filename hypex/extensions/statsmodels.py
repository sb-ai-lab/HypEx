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

    @staticmethod
    def _index_parts(index) -> tuple[list[str], list[str], list[str]]:
        """Split the ids of the p-values into test, field and group labels.

        An id is ``test<sep>params<sep>field`` and, when the p-value belongs to a
        particular test group, ``<sep>group`` on top of that.
        """
        parts = [str(i).split(ID_SPLIT_SYMBOL) for i in index]
        tests = [part[0] for part in parts]
        fields = [part[2] if len(part) > 2 else "" for part in parts]
        groups = [part[3] if len(part) > 3 else "" for part in parts]
        return tests, fields, groups

    def _calc_pandas(self, data: Dataset, **kwargs):
        p_values = data.data.values.flatten()
        tests, fields, groups = self._index_parts(data.index)

        corrected = np.empty(len(p_values), dtype=float)
        rejected = np.empty(len(p_values), dtype=bool)
        # every statistical test is a family of its own: the same metric checked
        # by a t-test and by a u-test must not inflate the correction of the other
        for test in dict.fromkeys(tests):
            positions = [i for i, name in enumerate(tests) if name == test]
            test_rejected, test_corrected = multipletests(
                [p_values[i] for i in positions],
                method=self.method.value,
                alpha=self.alpha,
                **kwargs,
            )[:2]
            corrected[positions] = test_corrected
            rejected[positions] = test_rejected

        return DatasetAdapter.to_dataset(
            {
                "field": fields,
                "group": groups,
                "test": tests,
                "old p-value": p_values,
                "new p-value": corrected,
                "correction": [
                    old / new if old != 0 else 0.0
                    for new, old in zip(corrected, p_values)
                ],
                "rejected": rejected,
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
