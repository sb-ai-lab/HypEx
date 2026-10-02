"""UI output handlers for A/B test results including CUPED and CUPAC."""

from __future__ import annotations

from ..analyzers.ab import ABAnalyzer
from ..comparators import GroupDifference, GroupSizes
from ..dataset import (
    Dataset,
    ExperimentData,
    SmallDataset,
    StatisticRole,
    TreatmentRole,
)
from ..reporters.ab import ABTestReporter, DictReporter
from ..reporters.abstract import _get_index_values
from ..reporters.cuped import CupedReporter
from ..utils import ID_SPLIT_SYMBOL, NAME_BORDER_SYMBOL, ExperimentDataEnum
from .base import Output

# ── CupedOutput ──────────────────────────────────────────────────────────────


class CupedOutput:
    """Container for CUPED variance-reduction results.

    Attributes:
        variance_reductions: Per-target variance reduction percentages.
    """

    def __init__(self) -> None:
        self.variance_reductions: SmallDataset | None = None

    def extract(self, experiment_data: ExperimentData) -> None:
        """Populate CUPED outputs from experiment data.

        Args:
            experiment_data: The experiment data container.
        """
        report = CupedReporter().report(experiment_data)
        if report.is_empty():
            self.variance_reductions = None
        else:
            self.variance_reductions = report

    def __repr__(self) -> str:
        if self.variance_reductions is None:
            return "CupedOutput(no CUPED data available)"
        return (
            f"CupedOutput(variance_reductions: "
            f"{len(self.variance_reductions)} target(s))"
        )


# ── CupacOutput ──────────────────────────────────────────────────────────────


class CupacOutput:
    """Container for CUPAC variance-reduction results.

    Attributes:
        variance_reductions: Per-target model selection and VR metrics.
        feature_importances: Per-(target, feature) importance scores.
    """

    def __init__(self) -> None:
        self.variance_reductions: Dataset | None = None
        self.feature_importances: Dataset | None = None

    def __repr__(self) -> str:
        has_vr = self.variance_reductions is not None
        has_fi = self.feature_importances is not None
        if not has_vr and not has_fi:
            return "CupacOutput(no CUPAC data available)"
        parts: list[str] = []
        if has_vr:
            parts.append(
                f"variance_reductions: {len(self.variance_reductions)} target(s)"
            )
        if has_fi:
            parts.append(f"feature_importances: {len(self.feature_importances)} rows")
        return f"CupacOutput({', '.join(parts)})"


# ── ABOutput ─────────────────────────────────────────────────────────────────


class ABOutput(Output):
    """Output handler for A/B test results.

    Attributes:
        multitest: Multiple-testing correction results or a message.
        sizes: Group size comparison table.
        cuped: CUPED variance-reduction outputs (when enabled).
        cupac: CUPAC variance-reduction outputs (when enabled).
    """

    multitest: Dataset | str
    sizes: Dataset
    cuped: CupedOutput | None
    cupac: CupacOutput

    def __init__(
        self,
        enable_cuped: bool = False,
        enable_cupac: bool = False,
    ) -> None:
        """Initialize AB test output handler.

        Args:
            enable_cuped: Whether CUPED was applied in the pipeline.
            enable_cupac: Whether CUPAC was applied in the pipeline.
        """
        self._groups: list[str] = []
        self.cuped = CupedOutput() if enable_cuped else None
        self.cupac = CupacOutput()
        super().__init__(
            summary_reporter=ABTestReporter(
                dict_reporter=DictReporter(),
                output_format="dataset",
                invert_pass=True,
            )
        )

    # ── Multitest ────────────────────────────────────────────────────

    def _extract_multitest_result(self, experiment_data: ExperimentData) -> None:
        """Extract multiple testing correction results from analysis tables.

        The correction is applied when the total number of comparisons
        ``(num_groups - 1) * num_target_fields`` exceeds 1 AND a
        correction method is configured.
        """
        multitest_id = experiment_data.get_one_id(
            ABAnalyzer,
            ExperimentDataEnum.analysis_tables,
        )
        if multitest_id and "MultiTest" in multitest_id:
            self.multitest = experiment_data.analysis_tables[multitest_id]
        else:
            self.multitest = (
                "Multiple testing correction was not applied: total "
                "comparisons ((groups-1) × targets) ≤ 1 or "
                "multitest_method was not provided."
            )

    # ── Differences ──────────────────────────────────────────────────

    def _extract_differences(self, experiment_data: ExperimentData) -> Dataset | None:
        targets: list[str] = []
        groups: list[str] = []
        ids = experiment_data.get_ids(
            GroupDifference,
            searched_space=ExperimentDataEnum.analysis_tables,
        )["GroupDifference"]["analysis_tables"]

        self._groups = [
            str(g)
            for g in list(
                experiment_data.groups[
                    experiment_data.ds.search_columns(TreatmentRole())[0]
                ].keys()
            )[1:]
        ]
        for i in self._groups:
            groups += [i] * len(ids)

        if not ids:
            return None

        diff = experiment_data.analysis_tables[ids[0]]
        for i in range(1, len(ids)):
            diff = diff.append(experiment_data.analysis_tables[ids[i]])
        for cid in ids:
            targets.append(cid.split(ID_SPLIT_SYMBOL)[-1])

        return diff.add_column(groups, role={"group": StatisticRole()}).add_column(
            targets * len(self._groups),
            role={"feature": StatisticRole()},
        )

    # ── Sizes ────────────────────────────────────────────────────────

    def _extract_sizes(self, experiment_data: ExperimentData) -> None:
        ids = experiment_data.get_ids(
            GroupSizes,
            searched_space=ExperimentDataEnum.analysis_tables,
        )["GroupSizes"]["analysis_tables"]
        main_ids = [i for i in ids if not i.endswith(f"{NAME_BORDER_SYMBOL}stats")]
        if not main_ids:
            main_ids = ids

        table = experiment_data.analysis_tables[main_ids[0]]
        index_values = _get_index_values(table)
        new_index: list[str] = []
        for idx in index_values:
            idx_str = str(idx)
            if NAME_BORDER_SYMBOL in idx_str:
                idx_str = idx_str.split(NAME_BORDER_SYMBOL)[0]
            try:
                idx_str = str(int(float(idx_str)))
            except (ValueError, TypeError):
                pass
            new_index.append(idx_str)
        table.index = new_index
        self.sizes = table.add_column(
            self._groups,
            role={"group": StatisticRole()},
        )

    # ── Variance reduction report property ───────────────────────────

    @property
    def variance_reduction_report(self) -> Dataset | str:
        """Get variance reduction report for CUPED/CUPAC transformations.

        Returns:
            A ``SmallDataset`` with variance reduction percentages per
            transformed metric, or a descriptive string if unavailable.
        """
        if hasattr(self, "_experiment_data"):
            return self.summary_reporter.report_variance_reductions(
                self._experiment_data,
            )
        return "No experiment data available."

    def extract(self, experiment_data: ExperimentData) -> None:
        """Extract all A/B test outputs including CUPED/CUPAC.

        Args:
            experiment_data: The experiment data container.
        """
        super().extract(experiment_data)
        self._extract_differences(experiment_data)
        self._extract_multitest_result(experiment_data)
        self._extract_sizes(experiment_data)
        self._extract_cupac(experiment_data)
        if self.cuped is not None:
            self.cuped.extract(experiment_data)

    def _extract_cupac(self, experiment_data: ExperimentData) -> None:
        """Delegate CUPAC extraction to CupacReporter (DRY).

        Args:
            experiment_data: The experiment data container.
        """
        from ..reporters.cupac import CupacReporter

        report = CupacReporter().report(experiment_data)
        self.cupac.variance_reductions = report.get("variance_reductions")
        self.cupac.feature_importances = report.get("feature_importances")
