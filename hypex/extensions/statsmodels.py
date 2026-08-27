from __future__ import annotations

import numpy as np
from scipy.stats import norm  # type: ignore
from statsmodels.stats.multitest import multipletests  # type: ignore

from ..dataset import Dataset, DatasetAdapter, StatisticRole
from ..utils import ID_SPLIT_SYMBOL, ABNTestMethodsEnum, BackendsEnum
from .abstract import Extension


class MultiTest(Extension):
    """Applies multiple testing correction to a collection of p-values.

    Wraps ``statsmodels.stats.multitest.multipletests`` and exposes it
    through the HypEx ``Extension`` interface so that both Pandas and
    Spark backends are supported transparently.

    Attributes:
        method: The correction method (e.g. ``holm``, ``bonferroni``).
        alpha: Family-wise error rate. Defaults to ``0.05``.
    """

    def __init__(self, method: ABNTestMethodsEnum, alpha: float = 0.05):
        self.method = method
        self.alpha = alpha
        super().__init__()

    def calc(self, data: Dataset, **kwargs):
        if data.backend_type == BackendsEnum.spark:
            return self._calc_spark(data, **kwargs)
        return self._calc_pandas(data, **kwargs)

    def _calc_pandas(self, data: Dataset, **kwargs):
        """Apply multiple-testing correction on a Pandas-backed dataset.

        The index of *data* is expected to carry composite identifiers
        (``ClassName┆params_hash┆field``) so that the corrected results
        can be mapped back to individual tests and target fields.  When
        the index contains non-string values (e.g. integers produced by
        ``pd.concat`` on an empty ``SmallDataset``), each element is
        safely converted to ``str`` before parsing.

        Args:
            data: A ``Dataset`` whose single column holds raw p-values
                and whose index encodes the test/field identifiers.
            **kwargs: Extra keyword arguments forwarded to
                ``statsmodels.stats.multitest.multipletests``.

        Returns:
            A ``Dataset`` with columns ``field``, ``test``,
            ``old p-value``, ``new p-value``, ``correction``, and
            ``rejected``.
        """
        p_values = data.data.values.flatten()
        new_pvalues = multipletests(
            p_values, method=self.method.value, alpha=self.alpha, **kwargs
        )

        fields: list[str] = []
        tests: list[str] = []
        for idx in data.index:
            s = str(idx)
            parts = s.split(ID_SPLIT_SYMBOL)
            fields.append(parts[2] if len(parts) > 2 else s)
            tests.append(parts[0] if len(parts) > 0 else s)

        return DatasetAdapter.to_dataset(
            {
                "field": fields,
                "test": tests,
                "old p-value": p_values,
                "new p-value": new_pvalues[1],
                "correction": [
                    j / i if i != 0 else 0.0
                    for i, j in zip(new_pvalues[1], p_values)
                ],
                "rejected": new_pvalues[0],
            },
            StatisticRole(),
        )

    def _calc_spark(self, data: Dataset, **kwargs):
        """Delegate to the Pandas implementation.

        Multiple-testing correction operates on a small, already-collected
        array of p-values (one per test × group), so converting to Pandas
        on the driver is safe and avoids reimplementing statsmodels logic
        in Spark.

        Args:
            data: A Spark-backed ``Dataset`` with raw p-values.
            **kwargs: Forwarded to ``_calc_pandas``.

        Returns:
            Corrected ``Dataset`` (Pandas-backed).
        """
        pdf = data.data.toPandas() if hasattr(data.data, "toPandas") else data.data
        pandas_ds = Dataset(roles=data.roles, data=pdf)
        return self._calc_pandas(pandas_ds, **kwargs)


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
        
    def _calc_spark(self, data: Dataset, **kwargs):
        """Delegates to the Pandas implementation.

        Quantile-based multitest runs Monte Carlo simulation on the driver,
        so data must already be small. Converting to Pandas is acceptable.
        """
        pdf = data.data.toPandas() if hasattr(data.data, "toPandas") else data.data
        pandas_ds = Dataset(
            roles=data.roles,
            data=pdf,
        )
        return self._calc_pandas(pandas_ds, **kwargs)

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
