from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..comparators import (
    GroupKSTest,
    GroupTTest,
    GroupUTest,
    StatsChi2Test,
    StatsKSTest,
    StatsTTest,
)
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
from ..utils.constants import ID_SPLIT_SYMBOL, NAME_BORDER_SYMBOL


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

        Retrieves treatment and target fields from the experiment data and calculates
        the total number of statistical comparisons being made. The correction is
        applied if the total number of comparisons (calculated as
        ``(num_groups - 1) * num_target_fields``) is strictly greater than 1.

        For standard correction methods (e.g., Bonferroni, Holm), it uses the
        ``MultiTest`` extension wrapping ``statsmodels``. For the ``quantile`` method,
        it uses the simulation-based ``MultitestQuantile`` extension.

        Args:
            data: The experiment data container holding dataset roles, groups,
                and metadata.
            p_values: A dataset containing the raw, uncorrected p-values to be
                adjusted.
            **kwargs: Extra keyword arguments forwarded to the underlying
                multitest extensions.

        Returns:
            ExperimentData: The updated experiment data instance with the
            multitest correction results stored in the analysis tables.
        """
        group_field = data.ds.search_columns(TreatmentRole())[0]
        target_fields = data.ds.search_columns(TargetRole(), search_types=[int, float])

        num_groups = len(data.groups[group_field])
        num_comparisons = (num_groups - 1) * len(target_fields)

        if self.multitest_method and num_comparisons > 1:
            if self.multitest_method != ABNTestMethodsEnum.quantile:
                multitest_result = MultiTest(self.multitest_method, self.alpha).calc(p_values, **kwargs)
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

    @staticmethod
    def _get_index_values(table: Dataset | SmallDataset) -> list[Any]:
        """Extract index values from a dataset in a backend-agnostic way.

        Args:
            table: The dataset instance.

        Returns:
            A list of index values.
        """
        index_obj = table.data.index
        if hasattr(index_obj, "to_list"):
            return index_obj.to_list()
        if hasattr(index_obj, "tolist"):
            return index_obj.tolist()
        return list(index_obj)

    @staticmethod
    def _extract_id_prefix(analysis_id: str) -> str:
        """Extract the ``ClassName┆params_hash`` prefix from a composite ID.

        Strips the trailing key portion (which, for vector executors,
        contains the full list of target column names) so that the
        resulting index follows the ``Test┆hash┆field┆group`` format
        expected by :meth:`MultiTest._index_parts`.

        Args:
            analysis_id: Full composite analysis-table ID.

        Returns:
            The first two ``ID_SPLIT_SYMBOL``-separated parts.
        """
        parts = analysis_id.split(ID_SPLIT_SYMBOL)
        if len(parts) >= 2:
            return ID_SPLIT_SYMBOL.join(parts[:2])
        return analysis_id

    def _build_row_index(
        self,
        t_data: Dataset | SmallDataset,
        analysis_ids: list[str],
        num_groups: int,
        group_labels: list[str],
    ) -> list[str]:
        """Build the composite row index for the aggregated test results.

        The resulting index format is::

            TestName┆params_hash┆target_field┆group_label

        which is the format expected by
        :meth:`~hypex.extensions.statsmodels.MultiTest._index_parts`.

        Handles two layouts:

        * **Iterative** (``len(analysis_ids) == num_targets``): one
          analysis_id per target; rows are target-major.
        * **Vector** (``len(analysis_ids) == 1`` and ``num_targets > 1``):
          a single ``StatsComparator`` execution produced rows for all
          targets; target names are parsed from the existing row index
          set by ``StatsComparator.execute()``.

        Args:
            t_data: The merged analysis table.
            analysis_ids: List of analysis table IDs for this test class.
            num_groups: Number of test groups (excluding baseline).
            group_labels: Ordered labels of test groups.

        Returns:
            A list of composite index strings, one per row in ``t_data``.
        """
        num_targets = len(t_data) // num_groups if num_groups > 0 else 1

        if len(analysis_ids) == num_targets:
            # Iterative mode: one analysis_id per target, rows are
            # target-major.  Each analysis_id already has the format
            # ``TestName┆hash┆single_target`` so we just append the group.
            return [
                f"{aid}{ID_SPLIT_SYMBOL}{group}"
                for aid in analysis_ids
                for group in group_labels
            ]

        # ── Vector mode ────────────────────────────────────────────────
        # A single analysis_id covers multiple targets.  The existing
        # index (set by StatsComparator) has the format
        # ``"{group}┆{col}"`` per row.  We parse target names from it
        # and build the correct composite index.
        existing_index = self._get_index_values(t_data)
        # Strip the key (which contains the list of ALL targets) and
        # keep only ``ClassName┆params_hash``.
        prefix = self._extract_id_prefix(analysis_ids[0])

        new_index: list[str] = []
        for idx_val in existing_index:
            idx_str = str(idx_val)
            if NAME_BORDER_SYMBOL in idx_str:
                group_part, target_part = idx_str.split(NAME_BORDER_SYMBOL, 1)
            else:
                group_part = idx_str
                target_part = ""
            new_index.append(
                f"{prefix}{ID_SPLIT_SYMBOL}{target_part}{ID_SPLIT_SYMBOL}{group_part}"
            )
        return new_index

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
        executor_ids = data.get_ids(
            [GroupTTest, GroupUTest, GroupKSTest,
             StatsTTest, StatsChi2Test, StatsKSTest]
        )

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
            analysis_ids = [
                aid for aid in analysis_ids
                if not aid.endswith(f"{NAME_BORDER_SYMBOL}stats")
            ]
            if len(analysis_ids) == 0:
                continue

            t_data = deepcopy(data.analysis_tables[analysis_ids[0]])
            for aid in analysis_ids[1:]:
                t_data = t_data.append(data.analysis_tables[aid])

            if len(t_data) > 0:
                group_labels = [groups[i][0] for i in range(1, num_groups + 1)]

                # ── Validation ──────────────────────────────────────────────
                # For vector executors (StatsComparator), a single analysis_id
                # contains rows for ALL targets × groups.  The correct check is
                # that total rows are divisible by num_groups.
                if num_groups > 0 and len(t_data) % num_groups != 0:
                    raise ValueError(
                        f"{c} produced {len(t_data)} rows which is not "
                        f"divisible by {num_groups} test group(s): the rows "
                        f"cannot be attributed to a target and a group."
                    )

                # ── Build row index ─────────────────────────────────────────
                row_index = self._build_row_index(
                    t_data, analysis_ids, num_groups, group_labels
                )
                t_data.data.index = row_index

                # ── Aggregate per-group statistics ──────────────────────────
                # Group rows by the trailing group label parsed from the
                # composite index (Test┆hash┆target┆group) instead of
                # slicing by position.  This is correct regardless of
                # whether rows are target-major or group-major.
                index_values = self._get_index_values(t_data)
                group_positions: dict[str, list[int]] = {
                    str(g[0]): [] for g in groups[1:]
                }
                for pos, idx_val in enumerate(index_values):
                    parts = str(idx_val).split(ID_SPLIT_SYMBOL)
                    grp_label = parts[-1] if len(parts) > 3 else ""
                    if grp_label in group_positions:
                        group_positions[grp_label].append(pos)

                for f in ["p-value", "pass"]:
                    all_positions = list(range(len(t_data)))
                    value_all = t_data.iloc[all_positions][f]
                    multitest_pvalues = self._add_pvalues(
                        multitest_pvalues, value_all, f
                    )
                    for grp_label, positions in group_positions.items():
                        if not positions:
                            continue
                        value = t_data.iloc[positions][f]
                        analysis_data[
                            f"{c} {f} {grp_label}"
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