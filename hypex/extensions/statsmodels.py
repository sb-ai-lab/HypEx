from __future__ import annotations

import numpy as np
from scipy.stats import norm  # type: ignore
from statsmodels.stats.multitest import multipletests  # type: ignore

from ..dataset import Dataset, DatasetAdapter, InfoRole, StatisticRole
from ..utils import ID_SPLIT_SYMBOL, ABNTestMethodsEnum, BackendsEnum
from ..utils.constants import TEST_NAME_NORMALIZATION
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

    @staticmethod
    def _index_parts(index) -> tuple[list[str], list[str], list[str]]:
        """Split composite p-value IDs into test, field and group labels.

        An id is ``test<sep>params<sep>field`` and, when the p-value
        belongs to a particular test group, ``<sep>group`` on top.
        """
        parts = [str(i).split(ID_SPLIT_SYMBOL) for i in index]
        tests = [part[0] for part in parts]
        fields = [part[2] if len(part) > 2 else "" for part in parts]
        groups = [part[3] if len(part) > 3 else "" for part in parts]
        return tests, fields, groups

    def _calc_pandas(self, data: Dataset, **kwargs):
        """Apply multiple testing correction to a Pandas-backed collection of p-values.

        Parses the composite index of *data* to identify which statistical test
        family (e.g. TTest, KSTest, Chi2Test) each p-value belongs to, then
        applies ``statsmodels.stats.multitest.multipletests`` **independently
        within each family**.  This ensures that corrections such as Holm or
        Bonferroni control the family-wise error rate per test type rather
        than across all heterogeneous comparisons simultaneously.

        The workflow is:
        1. Flatten the p-value matrix into a 1-D array.
        2. Decompose each index label into ``(test, field, group)`` via
           :meth:`_index_parts`.
        3. Normalize raw test class names (e.g. ``StatsTTest`` → ``TTest``)
           using :data:`~hypex.utils.constants.TEST_NAME_NORMALIZATION`.
        4. For every unique test family, collect the corresponding p-values
           and call ``multipletests(..., method=self.method.value,
           alpha=self.alpha)``.
        5. Assemble the results into a :class:`Dataset` with one row per
           original p-value.

        Args:
            data: A Pandas-backed ``Dataset`` whose values are raw,
                uncorrected p-values.  The index must follow the composite
                format ``test<sep>params<sep>field[<sep>group]`` (see
                :data:`~hypex.utils.constants.ID_SPLIT_SYMBOL`).
            **kwargs: Additional keyword arguments forwarded directly to
                ``statsmodels.stats.multitest.multipletests`` (e.g.
                ``maxiter`` for iterative methods).

        Returns:
            Dataset: A new ``Dataset`` (via ``DatasetAdapter.to_dataset``)
            with the following columns, all assigned
            :class:`~hypex.dataset.StatisticRole`:

            - ``"field"`` – the metric / feature name extracted from the
              index.
            - ``"test"`` – the normalized test family name (e.g.
              ``"TTest"``).
            - ``"old p-value"`` – the original, uncorrected p-value.
            - ``"new p-value"`` – the p-value after correction.
            - ``"correction"`` – the ratio ``old / new`` (``0.0`` when the
              old p-value is zero).
            - ``"rejected"`` – boolean flag indicating whether the null
              hypothesis is rejected at ``self.alpha`` after correction.
            - ``"group"`` – the compared-group label extracted from the
              index (empty string when not applicable).

        Raises:
            ValueError: If ``data`` contains no p-values or the index
                format is incompatible with :meth:`_index_parts`.

        Example:
            .. code-block:: python

                multitest = MultiTest(method=ABNTestMethodsEnum.holm, alpha=0.05)
                corrected_ds = multitest._calc_pandas(p_value_dataset)
                print(corrected_ds[["test", "old p-value", "new p-value", "rejected"]])
        """
        p_values = data.raw_data.values.flatten()
        tests_raw, fields, groups = self._index_parts(data.index)

        # Normalize BEFORE grouping into families
        tests = [TEST_NAME_NORMALIZATION.get(t, t) for t in tests_raw]

        corrected = np.empty(len(p_values), dtype=float)
        rejected = np.empty(len(p_values), dtype=bool)

        # Correction per statistical test family
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
                "test": tests,
                "old p-value": p_values,
                "new p-value": corrected,
                "correction": [
                    old / new if old != 0 else 0.0
                    for new, old in zip(corrected, p_values)
                ],
                "rejected": rejected,
                "group": groups,
            },
            StatisticRole(),
        )

    def _calc_spark(self, data: Dataset, **kwargs):
        """Delegate to the Pandas implementation via to_backend().

        Multiple-testing correction operates on a small, already-collected
        array of p-values (one per test × group), so converting to Pandas
        on the driver is safe. The composite string index must be present on
        the Spark dataset (it is lost by ``Dataset(pd.DataFrame, spark)``).
        """
        pandas_ds = data.to_backend(BackendsEnum.pandas)
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

    def calc(self, data: Dataset, **kwargs):
        if data.backend_type == BackendsEnum.spark:
            return self._calc_spark(data, **kwargs)
        return self._calc_pandas(data, **kwargs)

    def _calc_pandas(self, data: Dataset, **kwargs):
        group_field = kwargs.get("group_field")
        target_field = kwargs.get("target_field")
        quantiles = kwargs.get("quantiles")
        num_samples = len(data.unique()[group_field])
        sample_size = len(data)
        grouped_data = list(data[[group_field, target_field]].groupby(group_field))
        means = [sample[1][target_field].agg("mean") for sample in grouped_data]
        variances = [
            sample[1][target_field].agg("var") * sample_size / (sample_size - 1)
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
                    {"field": InfoRole(str), "accepted hypothesis": StatisticRole(int)},
                )
        return DatasetAdapter.to_dataset(
            {"field": target_field, "accepted hypothesis": 0},
            {"field": InfoRole(str), "accepted hypothesis": StatisticRole(int)},
        )

    def _calc_spark(self, data: Dataset, **kwargs):
        """Not supported on Spark.

        The pandas implementation needs per-group mean and variance of the raw
        target, which pyspark.pandas groups cannot provide here, and collecting
        the raw data to the driver is not acceptable.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "MultitestQuantile is not supported on the Spark backend. "
            "Use the pandas backend or another multitest_method "
            "(e.g. 'holm', 'bonferroni')."
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
