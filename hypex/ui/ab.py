"""UI output handlers for A/B test results including CUPED and CUPAC."""
from __future__ import annotations

from typing import Any

from ..analyzers.ab import ABAnalyzer
from ..comparators import GroupDifference, GroupSizes
from ..dataset import (
    Dataset,
    ExperimentData,
    InfoRole,
    SmallDataset,
    StatisticRole,
    TreatmentRole,
)
from ..reporters.ab import ABDatasetReporter, ABTestReporter
from ..reporters.abstract import _get_index_values
from ..transformers.cuped import CUPEDTransformer
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
        report = ABTestReporter.report_variance_reductions(experiment_data)
        if isinstance(report, str):
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
            parts.append(
                f"feature_importances: {len(self.feature_importances)} rows"
            )
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
        super().__init__(resume_reporter=ABDatasetReporter())

    # ── Multitest ────────────────────────────────────────────────────

    def _extract_multitest_result(self, experiment_data: ExperimentData) -> None:
        multitest_id = experiment_data.get_one_id(
            ABAnalyzer, ExperimentDataEnum.analysis_tables,
        )
        if multitest_id and "MultiTest" in multitest_id:
            self.multitest = experiment_data.analysis_tables[multitest_id]
        else:
            self.multitest = (
                "There was less than three groups or multitest method wasn't provided"
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
            targets * len(self._groups), role={"feature": StatisticRole()},
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
            self._groups, role={"group": StatisticRole()},
        )

    # ── CUPAC variance reductions ────────────────────────────────────

    def _extract_variance_reductions(self, experiment_data: ExperimentData) -> None:
        """Extract CUPAC variance reduction data from analysis_tables.

        CUPACExecutor stores reports under keys like
        ``CUPACExecutor┆hash┆target`` (not ``*_cupac_report``).
        Each report is a SmallDataset with columns
        ``cupac_best_model``, ``cupac_variance_reduction_cv``,
        ``cupac_variance_reduction_real``.
        """
        from ..ml.cupac import CUPACExecutor

        ids = experiment_data.get_ids(
            CUPACExecutor,
            searched_space=ExperimentDataEnum.analysis_tables,
        )
        all_ids = ids.get(CUPACExecutor.__name__, {}).get(
            ExperimentDataEnum.analysis_tables.value, [],
        )
        # Filter out importance sub-reports
        main_ids = [i for i in all_ids if not i.endswith("importances")]

        if not main_ids:
            self.cupac.variance_reductions = None
            return

        variance_data: list[dict[str, Any]] = []
        for aid in main_ids:
            table = experiment_data.analysis_tables.get(aid)
            if table is None or table.is_empty():
                continue
            records = table.to_records()
            if not records:
                continue
            rec = records[0]
            # Extract target name from composite ID
            target_name = aid.split(ID_SPLIT_SYMBOL)[-1]
            variance_data.append({
                "target": target_name,
                "best_model": rec.get("cupac_best_model"),
                "variance_reduction_cv": rec.get("cupac_variance_reduction_cv"),
                "variance_reduction_real": rec.get("cupac_variance_reduction_real"),
            })

        if variance_data:
            self.cupac.variance_reductions = SmallDataset.from_dict(
                variance_data,
                roles={
                    "target": InfoRole(str),
                    "best_model": InfoRole(str),
                    "variance_reduction_cv": StatisticRole(float),
                    "variance_reduction_real": StatisticRole(float),
                },
            )
        else:
            self.cupac.variance_reductions = None

    # ── CUPAC feature importances ────────────────────────────────────

    def _extract_feature_importances(self, experiment_data: ExperimentData) -> None:
        """Extract CUPAC feature importances from analysis_tables.

        CUPACExecutor stores importances under keys like
        ``CUPACExecutor┆hash┆target┆importances``.
        """
        from ..ml.cupac import CUPACExecutor

        ids = experiment_data.get_ids(
            CUPACExecutor,
            searched_space=ExperimentDataEnum.analysis_tables,
        )
        all_ids = ids.get(CUPACExecutor.__name__, {}).get(
            ExperimentDataEnum.analysis_tables.value, [],
        )
        imp_ids = [i for i in all_ids if i.endswith("importances")]

        if not imp_ids:
            self.cupac.feature_importances = None
            return

        importance_data: list[dict[str, Any]] = []
        for aid in imp_ids:
            table = experiment_data.analysis_tables.get(aid)
            if table is None or table.is_empty():
                continue
            records = table.to_records()
            if not records:
                continue
            rec = records[0]
            # Extract target from ID: CUPACExecutor┆hash┆target┆importances
            parts = aid.split(ID_SPLIT_SYMBOL)
            target_name = parts[-2] if len(parts) >= 2 else "unknown"
            # Find model from the main report
            main_id = ID_SPLIT_SYMBOL.join(parts[:-1])
            main_table = experiment_data.analysis_tables.get(main_id)
            model_name = None
            if main_table and not main_table.is_empty():
                main_records = main_table.to_records()
                if main_records:
                    model_name = main_records[0].get("cupac_best_model")
            for feature, importance in rec.items():
                importance_data.append({
                    "target": target_name,
                    "feature": feature,
                    "importance": importance,
                    "model": model_name,
                })

        if importance_data:
            self.cupac.feature_importances = SmallDataset.from_dict(
                importance_data,
                roles={
                    "target": InfoRole(str),
                    "feature": InfoRole(str),
                    "importance": StatisticRole(float),
                    "model": InfoRole(str),
                },
            )
        else:
            self.cupac.feature_importances = None

    # ── Variance reduction report property ───────────────────────────

    @property
    def variance_reduction_report(self) -> Dataset | str:
        """Get variance reduction report for CUPED/CUPAC transformations.

        Returns:
            A ``SmallDataset`` with variance reduction percentages per
            transformed metric, or a descriptive string if unavailable.
        """
        if hasattr(self, "_experiment_data"):
            return self.resume_reporter.report_variance_reductions(
                self._experiment_data,
            )
        return "No experiment data available."

    # ── Main extract ─────────────────────────────────────────────────

    def extract(self, experiment_data: ExperimentData) -> None:
        """Extract all A/B test outputs including CUPED/CUPAC.

        Args:
            experiment_data: The experiment data container.
        """
        super().extract(experiment_data)
        self._extract_differences(experiment_data)
        self._extract_multitest_result(experiment_data)
        self._extract_sizes(experiment_data)
        self._extract_variance_reductions(experiment_data)
        self._extract_feature_importances(experiment_data)

        if self.cuped is not None:
            self.cuped.extract(experiment_data)
