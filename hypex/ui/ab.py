"""UI output handlers for A/B test results including CUPED and CUPAC."""

from __future__ import annotations

import html
from typing import Any

from ..analyzers.ab import ABAnalyzer
from ..comparators import GroupDifference, GroupSizes
from ..dataset import (
    Dataset,
    ExperimentData,
    SmallDataset,
    StatisticRole,
    TreatmentRole,
)
from ..reporters.ab import ABTestReporter
from ..reporters.abstract import DictReporter, _get_index_values
from ..reporters.cuped import CupedReporter
from ..utils import ID_SPLIT_SYMBOL, NAME_BORDER_SYMBOL, ExperimentDataEnum
from ..utils.logger import logger
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

    @staticmethod
    def _vr_color(pct: float) -> str:
        """Return a CSS color based on variance-reduction magnitude."""
        if pct >= 30:
            return "#2e7d32"  # green
        if pct >= 10:
            return "#f57f17"  # amber
        return "#c62828"  # red

    @staticmethod
    def _progress_bar(pct: float, width: int = 20) -> str:
        """Render a text progress bar."""
        filled = round(pct / 100 * width)
        filled = max(0, min(width, filled))
        return "█" * filled + "░" * (width - filled)

    def __repr__(self) -> str:
        if self.variance_reductions is None:
            return "CupedOutput(no CUPED data available)"

        lines = ["CupedOutput:"]
        lines.append(
            f"  variance_reductions ({len(self.variance_reductions)} target(s)):"
        )
        lines.append("")

        records = self.variance_reductions.to_records()
        for rec in records:
            feat = rec.get("feature", "?")
            pct = rec.get("variance_reduction_pct", 0)
            bar = self._progress_bar(pct)
            lines.append(f"    {feat:<20s} {bar} {pct:>6.1f}%")

        mean_pct = sum(r.get("variance_reduction_pct", 0) for r in records) / max(
            len(records), 1
        )
        lines.append("")
        lines.append(f"    Average VR: {mean_pct:.1f}%")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        if self.variance_reductions is None:
            return (
                "<div style='font-family:monospace;'>"
                "<b>CupedOutput:</b> no CUPED data available</div>"
            )

        records = self.variance_reductions.to_records()
        rows_html = ""
        for rec in records:
            feat = rec.get("feature", "?")
            pct = rec.get("variance_reduction_pct", 0)
            color = self._vr_color(pct)
            bar_width = max(0, min(100, pct))
            rows_html += (
                "<tr>"
                f"<td style='padding:2px 8px; font-family:monospace;'>{feat}</td>"
                "<td style='padding:2px 8px;'>"
                f"<div style='background:#e0e0e0; border-radius:3px; "
                f"width:120px; height:14px; display:inline-block; "
                f"vertical-align:middle;'>"
                f"<div style='background:{color}; height:100%; "
                f"width:{bar_width}%; border-radius:3px;'></div>"
                "</div>"
                "</td>"
                f"<td style='padding:2px 8px; font-family:monospace; "
                f"color:{color}; font-weight:bold;'>{pct:.1f}%</td>"
                "</tr>"
            )

        mean_pct = sum(r.get("variance_reduction_pct", 0) for r in records) / max(
            len(records), 1
        )

        return (
            "<div style='font-family:monospace; display:inline-block;'>"
            "<b>CupedOutput</b>"
            f"<span style='color:#555; margin-left:8px;'>"
            f"({len(self.variance_reductions)} target(s))</span>"
            "<table style='border-collapse:collapse; margin:6px 0;'>"
            f"<tr><th style='text-align:left; padding:2px 8px;'>Feature</th>"
            f"<th style='text-align:left; padding:2px 8px;'>Variance Reduction</th>"
            f"<th style='text-align:right; padding:2px 8px;'>%</th></tr>"
            f"{rows_html}"
            "</table>"
            f"<div style='margin-top:4px; color:#555; font-size:11px;'>"
            f"Average VR: <b>{mean_pct:.1f}%</b></div>"
            "</div>"
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

    @staticmethod
    def _vr_color(pct: float) -> str:
        if pct >= 30:
            return "#2e7d32"
        if pct >= 10:
            return "#f57f17"
        return "#c62828"

    @staticmethod
    def _text_bar(value: float, max_val: float, width: int = 15) -> str:
        """Render a horizontal text bar."""
        if max_val <= 0:
            return ""
        filled = round(value / max_val * width)
        filled = max(0, min(width, filled))
        return "▓" * filled + "░" * (width - filled)

    def __repr__(self) -> str:
        has_vr = self.variance_reductions is not None
        has_fi = self.feature_importances is not None
        if not has_vr and not has_fi:
            return "CupacOutput(no CUPAC data available)"

        parts: list[str] = ["CupacOutput:"]

        if has_vr:
            parts.append(
                f"  variance_reductions ({len(self.variance_reductions)} target(s)):"
            )
            records = self.variance_reductions.to_records()
            for rec in records:
                target = rec.get("target", "?")
                model = rec.get("best_model", "?")
                vr_cv = rec.get("variance_reduction_cv", 0)
                vr_real = rec.get("variance_reduction_real", 0)
                parts.append(
                    f"    {target:<15s} model={model:<10s} "
                    f"VR(cv)={vr_cv:>5.1f}%  VR(real)={vr_real:>5.1f}%"
                )

        if has_fi:
            parts.append(
                f"  feature_importances ({len(self.feature_importances)} rows):"
            )
            records = self.feature_importances.to_records()
            # Sort by importance descending
            sorted_recs = sorted(
                records,
                key=lambda r: r.get("importance", 0),
                reverse=True,
            )
            max_imp = max((r.get("importance", 0) for r in sorted_recs), default=1)
            for rec in sorted_recs[:10]:  # top-10
                feat = rec.get("feature", "?")
                imp = rec.get("importance", 0)
                bar = self._text_bar(imp, max_imp)
                parts.append(f"    {feat:<20s} {bar} {imp:>8.4f}")
            if len(sorted_recs) > 10:
                parts.append(f"    ... and {len(sorted_recs) - 10} more")

        return "\n".join(parts)

    def _repr_html_(self) -> str:
        has_vr = self.variance_reductions is not None
        has_fi = self.feature_importances is not None
        if not has_vr and not has_fi:
            return (
                "<div style='font-family:monospace;'>"
                "<b>CupacOutput:</b> no CUPAC data available</div>"
            )

        sections: list[str] = []

        if has_vr:
            records = self.variance_reductions.to_records()
            rows = ""
            for rec in records:
                target = rec.get("target", "?")
                model = rec.get("best_model", "?")
                vr_cv = rec.get("variance_reduction_cv", 0)
                vr_real = rec.get("variance_reduction_real", 0)
                color = self._vr_color(vr_real)
                rows += (
                    "<tr>"
                    f"<td style='padding:2px 8px;'>{target}</td>"
                    f"<td style='padding:2px 8px;'><code>{model}</code></td>"
                    f"<td style='padding:2px 8px; text-align:right;'>{vr_cv:.1f}%</td>"
                    f"<td style='padding:2px 8px; text-align:right; "
                    f"color:{color}; font-weight:bold;'>{vr_real:.1f}%</td>"
                    "</tr>"
                )
            sections.append(
                "<div style='margin:6px 0 2px 0; color:#555; font-size:12px;'>"
                f"variance_reductions ({len(self.variance_reductions)} target(s))</div>"
                "<table style='border-collapse:collapse; font-size:12px;'>"
                "<tr>"
                "<th style='text-align:left; padding:2px 8px;'>Target</th>"
                "<th style='text-align:left; padding:2px 8px;'>Model</th>"
                "<th style='text-align:right; padding:2px 8px;'>VR (CV)</th>"
                "<th style='text-align:right; padding:2px 8px;'>VR (real)</th>"
                "</tr>"
                f"{rows}</table>"
            )

        if has_fi:
            records = self.feature_importances.to_records()
            sorted_recs = sorted(
                records, key=lambda r: r.get("importance", 0), reverse=True
            )
            max_imp = max((r.get("importance", 0) for r in sorted_recs), default=1)
            rows = ""
            for rec in sorted_recs[:10]:
                feat = rec.get("feature", "?")
                imp = rec.get("importance", 0)
                bar_pct = max(0, min(100, imp / max_imp * 100))
                rows += (
                    "<tr>"
                    f"<td style='padding:2px 8px;'>{feat}</td>"
                    "<td style='padding:2px 8px;'>"
                    f"<div style='background:#e3f2fd; border-radius:2px; "
                    f"width:100px; height:12px; display:inline-block;'>"
                    f"<div style='background:#1565c0; height:100%; "
                    f"width:{bar_pct}%; border-radius:2px;'></div></div>"
                    "</td>"
                    f"<td style='padding:2px 8px; text-align:right; "
                    f"font-size:11px;'>{imp:.4f}</td>"
                    "</tr>"
                )
            if len(sorted_recs) > 10:
                rows += (
                    f"<tr><td colspan='3' style='padding:2px 8px; "
                    f"color:#999;'>… and {len(sorted_recs) - 10} more</td></tr>"
                )
            sections.append(
                "<div style='margin:8px 0 2px 0; color:#555; font-size:12px;'>"
                f"feature_importances ({len(self.feature_importances)} rows)</div>"
                "<table style='border-collapse:collapse; font-size:12px;'>"
                "<tr>"
                "<th style='text-align:left; padding:2px 8px;'>Feature</th>"
                "<th style='text-align:left; padding:2px 8px;'>Importance</th>"
                "<th style='text-align:right; padding:2px 8px;'>Value</th>"
                "</tr>"
                f"{rows}</table>"
            )

        return (
            "<div style='font-family:monospace; display:inline-block;'>"
            "<b>CupacOutput</b>" + "".join(sections) + "</div>"
        )


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
                # non-numeric index: keep the string as is
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

        CUPED results are returned when CUPED was applied, otherwise CUPAC
        results (per-target variance reduction of the selected models).
        When both are present the CUPAC table stays available as
        ``cupac.variance_reductions``.

        Returns:
            A ``Dataset`` with variance reduction percentages per transformed
            metric, or a descriptive string if unavailable.
        """
        if self.cuped is not None and self.cuped.variance_reductions is not None:
            return self.cuped.variance_reductions
        if self.cupac.variance_reductions is not None:
            return self.cupac.variance_reductions
        return "No variance reduction data available."

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

    def _summary_stats(self) -> dict[str, Any]:
        """Collect high-level AB statistics for the summary block."""
        stats: dict[str, Any] = {}

        # Feature / group counts from summary
        if hasattr(self, "summary") and self.summary is not None:
            try:
                recs = self.summary.to_records()
                features = set()
                groups = set()
                for r in recs:
                    if "feature" in r:
                        features.add(r["feature"])
                    if "group" in r:
                        groups.add(r["group"])
                stats["features"] = len(features)
                stats["groups"] = len(groups)
            except Exception:
                # best-effort: summary table may be empty or malformed
                logger.debug("ABOutput._summary_stats: failed to parse summary")

        # Sizes
        if hasattr(self, "sizes") and self.sizes is not None:
            try:
                recs = self.sizes.to_records()
                if recs:
                    stats["control_size"] = recs[0].get("control size")
                    stats["test_size"] = recs[0].get("test size")
            except Exception:
                # best-effort: sizes table may be empty or malformed
                logger.debug("ABOutput._summary_stats: failed to parse sizes")

        # Multitest method
        if hasattr(self, "multitest") and self.multitest is not None:
            if isinstance(self.multitest, str):
                stats["multitest"] = "not applied"
            else:
                try:
                    recs = self.multitest.to_records()
                    if recs and "correction" in recs[0]:
                        stats["multitest"] = "applied"
                except Exception:
                    # best-effort: multitest table may be empty or malformed
                    logger.debug("ABOutput._summary_stats: failed to parse multitest")

        # CUPED / CUPAC presence
        if self.cuped is not None:
            stats["cuped"] = self.cuped.variance_reductions is not None
        if self.cupac.variance_reductions is not None:
            stats["cupac"] = True

        return stats

    def _result_rows(self) -> list[dict[str, Any]]:
        """Extract per-feature pass/fail rows from summary."""
        rows: list[dict[str, Any]] = []
        if not hasattr(self, "summary") or self.summary is None:
            return rows
        try:
            recs = self.summary.to_records()
            for r in recs:
                feature = r.get("feature", "?")
                # Find any column ending with " pass"
                pass_val = None
                pval = None
                diff_pct = None
                for k, v in r.items():
                    if k.endswith(" pass"):
                        pass_val = v
                    elif k.endswith(" p-value"):
                        pval = v
                    elif k == "difference %":
                        diff_pct = v
                rows.append(
                    {
                        "feature": feature,
                        "pass": pass_val,
                        "p_value": pval,
                        "diff_pct": diff_pct,
                    }
                )
        except Exception:
            # best-effort: keep repr working even if summary is malformed
            logger.debug("ABOutput._result_rows: failed to parse summary rows")
        return rows

    def __repr__(self) -> str:
        s = self._summary_stats()
        lines = ["ABOutput"]

        meta: list[str] = []
        if "features" in s:
            meta.append(f"{s['features']} feature(s)")
        if "groups" in s:
            meta.append(f"{s['groups']} group(s)")
        if "control_size" in s and "test_size" in s:
            meta.append(f"sizes: {s['control_size']} / {s['test_size']}")
        if "multitest" in s:
            meta.append(f"multitest: {s['multitest']}")
        if s.get("cuped"):
            meta.append("CUPED ✓")
        if s.get("cupac"):
            meta.append("CUPAC ✓")
        if meta:
            lines.append(f"  {', '.join(meta)}")

        # Per-feature results
        rows = self._result_rows()
        if rows:
            lines.append("")
            for r in rows:
                icon = "✅" if r["pass"] == "OK" else "❌"
                pval_str = f"{r['p_value']:.4f}" if r["p_value"] is not None else "—"
                diff_str = f"{r['diff_pct']:+.1f}%" if r["diff_pct"] is not None else ""
                lines.append(f"  {icon} {r['feature']:<20s} p={pval_str}  {diff_str}")

        lines.append("")
        lines.append(repr(self._report()))
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        s = self._summary_stats()
        parts: list[str] = [
            "<div style='font-family:monospace; display:inline-block;'>",
            "<b style='font-size:14px;'>ABOutput</b>",
        ]

        # Meta row
        badges: list[str] = []
        if "features" in s:
            badges.append(f"{s['features']} feature(s)")
        if "groups" in s:
            badges.append(f"{s['groups']} group(s)")
        if "control_size" in s and "test_size" in s:
            badges.append(f"sizes: {s['control_size']} / {s['test_size']}")
        if s.get("cuped"):
            badges.append("CUPED ✓")
        if s.get("cupac"):
            badges.append("CUPAC ✓")
        if badges:
            parts.append(
                "<div style='margin:4px 0; color:#555; font-size:12px;'>"
                + " &nbsp;|&nbsp; ".join(html.escape(b) for b in badges)
                + "</div>"
            )

        # Per-feature result table
        rows = self._result_rows()
        if rows:
            parts.append(
                "<table style='border-collapse:collapse; margin:6px 0; font-size:12px;'>"
                "<tr>"
                "<th style='padding:2px 8px; text-align:left;'>Feature</th>"
                "<th style='padding:2px 8px; text-align:center;'>Result</th>"
                "<th style='padding:2px 8px; text-align:right;'>p-value</th>"
                "<th style='padding:2px 8px; text-align:right;'>Diff %</th>"
                "</tr>"
            )
            for r in rows:
                ok = r["pass"] == "OK"
                color = "#2e7d32" if ok else "#c62828"
                icon = "✅" if ok else "❌"
                pval = f"{r['p_value']:.4f}" if r["p_value"] is not None else "—"
                diff = f"{r['diff_pct']:+.1f}%" if r["diff_pct"] is not None else "—"
                parts.append(
                    "<tr>"
                    f"<td style='padding:2px 8px;'>{html.escape(str(r['feature']))}</td>"
                    f"<td style='padding:2px 8px; text-align:center; color:{color};'>{icon}</td>"
                    f"<td style='padding:2px 8px; text-align:right;'>{pval}</td>"
                    f"<td style='padding:2px 8px; text-align:right;'>{diff}</td>"
                    "</tr>"
                )
            parts.append("</table>")

        parts.append(self._report()._repr_html_())
        parts.append("</div>")
        return "".join(parts)
