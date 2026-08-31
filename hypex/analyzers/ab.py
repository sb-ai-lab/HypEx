from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..comparators import GroupTTest, GroupUTest, StatsChi2Test, StatsTTest
from ..dataset import (
    Dataset,
    ExperimentData,
    InfoRole,
    StatisticRole,
    TargetRole,
    TreatmentRole,
)
from ..dataset.dataset import SmallDataset
from ..experiments.base import Executor
from ..extensions.statsmodels import MultiTest, MultitestQuantile
from ..utils import ABNTestMethodsEnum, ExperimentDataEnum, timeit


class ABAnalyzer(Executor):
    """Analyzer for A/B test results with multiple testing correction support.

    Aggregates statistical test results (t-test, U-test, chi-square) from
    A/B experiments and applies multiple testing correction methods to control
    family-wise error rate or false discovery rate. Supports both standard
    corrections (Bonferroni, Holm, etc.) and quantile-based methods for
    multi-group comparisons.

    Attributes:
        multitest_method: Method for multiple testing correction.
        alpha: Significance level for hypothesis testing.
        equal_variance: Whether to assume equal variances in quantile method.
        quantiles: Pre-computed quantile thresholds for marginal distributions.
        iteration_size: Number of Monte Carlo iterations for quantile estimation.
        random_state: Random seed for reproducibility.
    """
    def __init__(
        self,
        multitest_method: ABNTestMethodsEnum | None = None,
        alpha: float = 0.05,
        equal_variance: bool = True,
        quantiles: float | list[float] | None = None,
        iteration_size: int = 20000,
        random_state: int | None = None,
        key: Any = "",
    ):
        """Initializes the A/B test analyzer.

        Args:
            multitest_method: Method for multiple testing correction.
                Options include ``bonferroni``, ``holm``, ``fdr_bh``, etc.
                If ``None``, no correction is applied.
            alpha: Significance level (Type I error rate) for hypothesis tests.
                Defaults to 0.05.
            equal_variance: Whether to assume equal variances across groups
                when using the quantile-based correction method. Defaults to ``True``.
            quantiles: Pre-computed critical quantile values for the marginal
                distribution of test statistics. If ``None``, computed internally.
            iteration_size: Number of Monte Carlo iterations for estimating
                quantiles of the marginal distribution. Defaults to 20000.
            random_state: Random seed for reproducibility of Monte Carlo sampling.
                If ``None``, results may vary between runs.
            key: Optional identifier key for storing results in experiment data.
        """
        self.multitest_method = multitest_method
        self.alpha = alpha
        self.equal_variance = equal_variance
        self.quantiles = quantiles
        self.iteration_size = iteration_size
        self.random_state = random_state
        super().__init__(key)

    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        """Stores a value in the experiment data's analysis tables.

        Args:
            data: The experiment data container to update.
            value: The value (typically a ``SmallDataset``) to store.
            key: Optional suffix to append to the executor ID for the storage key.

        Returns:
            The updated ``ExperimentData`` instance.
        """
        return data.set_value(
            ExperimentDataEnum.analysis_tables,
            self.id + key if key else self.id,
            value,
        )

    def execute_multitest(self, data: ExperimentData, p_values: Dataset, **kwargs):
        """Applies multiple testing correction to aggregated p-values.

        Retrieves treatment and target fields from the experiment data, then
        applies the specified correction method if more than two groups exist.
        For standard methods, uses ``MultiTest`` from statsmodels. For the
        ``quantile`` method, uses simulation-based ``MultitestQuantile``.

        Args:
            data: The experiment data container with group information.
            p_values: Dataset containing raw p-values from statistical tests.
            **kwargs: Additional arguments passed to the correction method.

        Returns:
            Updated ``ExperimentData`` with corrected p-values stored under
            the ``"MultiTest"`` key, or the original ``data`` if correction
            is not applicable.
        """
        group_field = data.ds.search_columns(TreatmentRole())[0]
        target_fields = data.ds.search_columns(TargetRole(), search_types=[int, float])
        if self.multitest_method and len(data.groups[group_field]) > 2:
            if self.multitest_method != ABNTestMethodsEnum.quantile:
                multitest_result = MultiTest(self.multitest_method).calc(
                    p_values, **kwargs
                )
                groups = []
                for i in list(data.groups[group_field].keys())[1:]:
                    groups += [i] * len(target_fields)
                multitest_result = multitest_result.add_column(
                    groups
                    * (
                        len(multitest_result)
                        // len(target_fields)
                        // (len(data.groups[group_field]) - 1)
                    ),
                    role={"group": StatisticRole()},
                )

            else:
                multitest_result = SmallDataset.create_empty()
                for target_field in target_fields:
                    multitest_result = multitest_result.append(
                        MultitestQuantile(
                            self.alpha,
                            self.iteration_size,
                            self.equal_variance,
                            self.random_state,
                        ).calc(
                            p_values,
                            group_field=group_field,
                            target_field=target_field,
                            quantiles=self.quantiles,
                        )
                    )
            return self._set_value(data, multitest_result, key="MultiTest")
        return data

    def _add_pvalues(self, multitest_pvalues, value, field):
        """Conditionally appends p-values for multiple testing correction.

        Adds p-values to the collection only if a correction method is specified,
        the field is ``"p-value"``, and the method is not the quantile-based approach
        (which handles p-values differently).

        Args:
            multitest_pvalues: The accumulating dataset of p-values.
            value: The p-value dataset or column to potentially append.
            field: The field name being processed (e.g., ``"p-value"`` or ``"pass"``).

        Returns:
            The updated ``multitest_pvalues`` dataset.
        """
        if (
            self.multitest_method
            and field == "p-value"
            and self.multitest_method != "quantile"
        ):
            multitest_pvalues = multitest_pvalues.append(value)
        return multitest_pvalues

    @timeit(level="ANALYZER", prefix="AB_ANALYZER")
    def execute(self, data: ExperimentData) -> ExperimentData:
        """Executes the full A/B test analysis pipeline.

        Aggregates results from registered statistical tests (t-test, U-test,
        chi-square, etc.), computes mean p-values and pass rates across groups,
        applies multiple testing correction if configured, and stores the
        aggregated metrics in the experiment data.

        The method handles:
            1. Retrieving test results by executor class from ``ExperimentData``.
            2. Populating ``data.groups`` when absent (Spark/StatsComparator path
            does not populate groups, unlike GroupsComparator on Pandas).
            3. Merging results from multiple iterations or groups.
            4. Computing aggregate statistics (mean p-value, pass rate) per test.
            5. Applying multiple testing correction via ``execute_multitest``.
            6. Storing the final analysis dataset in ``analysis_tables``.

        Args:
            data: The ``ExperimentData`` container with test results.

        Returns:
            Updated ``ExperimentData`` with aggregated analysis results stored
            under the analyzer's executor ID.

        Raises:
            KeyError: If the treatment column cannot be found in the dataset.
        """
        executor_ids = data.get_ids([GroupTTest, GroupUTest, StatsTTest, StatsChi2Test])

        group_field = data.ds.search_columns(TreatmentRole())[0]

        # StatsComparator-based executors (used on Spark backend) do NOT populate
        # data.groups, unlike GroupsComparator (used on Pandas backend).
        # Populate groups here if they are missing to ensure downstream logic
        # (num_groups, group labels, multitest) works correctly.
        if group_field not in data.groups:
            combined_data = data.ds
            if group_field in combined_data.columns:
                inner_df = (
                    combined_data.data
                    if hasattr(combined_data, "data")
                    else combined_data.backend_data.data
                )
                initial_len = len(inner_df)
                inner_df = inner_df.dropna(subset=[group_field])
                dropped = initial_len - len(inner_df)
                if dropped > 0:
                    combined_data = type(combined_data)(
                        data=inner_df,
                        roles={
                            c: combined_data.roles.get(c, InfoRole())
                            for c in inner_df.columns
                        },
                    )
                data.groups[group_field] = {
                    f"{group}": ds
                    for group, ds in combined_data.groupby(group_field)
                }

        num_groups = len(data.groups[group_field]) - 1
        groups = list(data.groups[group_field].items())
        multitest_pvalues = SmallDataset.create_empty()
        analysis_data = {}

        for c, spaces in executor_ids.items():
            analysis_ids = spaces.get("analysis_tables", [])
            if len(analysis_ids) == 0:
                continue

            t_data = deepcopy(data.analysis_tables[analysis_ids[0]])
            for aid in analysis_ids[1:]:
                t_data = t_data.append(data.analysis_tables[aid])

            if len(t_data) > 0:
                current_index_len = (
                    len(t_data.data.index)
                    if hasattr(t_data.data, "index")
                    else 0
                )
                if current_index_len != len(t_data):
                    new_index = []
                    for i in range(len(t_data)):
                        if i < len(analysis_ids):
                            new_index.append(analysis_ids[i])
                        else:
                            col_name = (
                                t_data.columns[0]
                                if len(t_data.columns) > 0
                                else "metric"
                            )
                            new_index.append(f"{c}┴┴{col_name}┴┴row{i}")
                    t_data.data.index = new_index

                for f in ["p-value", "pass"]:
                    step = len(t_data) // num_groups if num_groups > 0 else 1
                    if step == 0:
                        step = 1
                    
                    for i in range(0, len(t_data), step):
                        slice_start = i
                        slice_end = min(i + step, len(t_data))
                        
                        value = t_data.iloc[slice_start:slice_end][f]
                        multitest_pvalues = self._add_pvalues(
                            multitest_pvalues, value, f
                        )
                        
                        group_idx = i // step + 1
                        if group_idx >= len(groups):
                            group_idx = len(groups) - 1

                        analysis_data[
                            f"{c} {f} {groups[group_idx][0]}"
                        ] = value.mean()

        analysis_dataset = SmallDataset.from_dict(
            [analysis_data], {f: StatisticRole(float) for f in analysis_data}
        )

        data = self.execute_multitest(
            data,
            (
                multitest_pvalues
                if not multitest_pvalues.is_empty()
                and self.multitest_method != ABNTestMethodsEnum.quantile
                else data.ds
            ),
        )

        return self._set_value(data, analysis_dataset)
