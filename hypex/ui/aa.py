import html
import math
from typing import Any

from ..analyzers.aa import AADryTestAnalyzer, AAScoreAnalyzer
from ..dataset import Dataset, ExperimentData, InfoRole, SmallDataset, StatisticRole
from ..reporters.aa import AABestSplitReporter, AAPassedReporter
from ..utils import ExperimentDataEnum, _parse_metric_col
from ..utils.constants import (
    ID_SPLIT_SYMBOL,
    NAME_BORDER_SYMBOL,
    TEST_NAME_NORMALIZATION,
)
from ..utils.enums import RenameEnum
from ..utils.naming import normalize_test_name
from .base import Output


class AAOutput(Output):
    best_split: Dataset
    experiments: Dataset
    aa_score: Dataset
    best_split_statistics: Dataset

    def __init__(self):
        super().__init__(
            summary_reporter=AAPassedReporter(),
            additional_reporters={"best_split": AABestSplitReporter()},
        )

    def _extract_best_split_statistics(self, experiment_data: ExperimentData):
        aa_score_analyser_ids = experiment_data.get_ids(
            AAScoreAnalyzer, ExperimentDataEnum.analysis_tables
        )[AAScoreAnalyzer.__name__][ExperimentDataEnum.analysis_tables.value]

        best_split_id = None
        for id_ in aa_score_analyser_ids:
            if id_.endswith("best split statistics"):
                best_split_id = id_
                break

        if best_split_id is None:
            self.best_split_statistics = SmallDataset.create_empty()
            return

        raw_table = experiment_data.analysis_tables[best_split_id]
        if raw_table.is_empty():
            self.best_split_statistics = SmallDataset.create_empty()
            return

        records = raw_table.to_records()
        if not records:
            self.best_split_statistics = SmallDataset.create_empty()
            return

        row = records[0]

        feature_groups: set[tuple[str, str]] = set()
        test_names: set[str] = set()

        for k in row.keys():
            if NAME_BORDER_SYMBOL in k:
                continue
            feature, test, _metric, group = _parse_metric_col(k)
            if feature and feature != "mean":
                feature_groups.add((feature, group))
                if test and test != "GroupDifference":
                    test_names.add(normalize_test_name(test))

        order_map = {"TTest": 0, "KSTest": 1, "Chi2Test": 2, "UTest": 3, "ZTest": 4}
        ordered_tests = sorted(test_names, key=lambda x: order_map.get(x, 99))

        result_rows = []
        for feature, group in sorted(feature_groups):
            rec = {
                "feature": feature,
                "group": group,
                "control mean": None,
                "test mean": None,
                "difference": None,
                "difference %": None,
            }

            for k, v in row.items():
                f, t, m, g = _parse_metric_col(k)
                if f == feature and g == group and t == "GroupDifference":
                    if m == "control mean":
                        rec["control mean"] = v
                    elif m == "test mean":
                        rec["test mean"] = v
                    elif m == "difference":
                        rec["difference"] = v
                    elif m == "difference %":
                        rec["difference %"] = v

            for tn in ordered_tests:
                rec[f"{tn} pass"] = None
                rec[f"{tn} p-value"] = None
                for k, v in row.items():
                    f, t, m, g = _parse_metric_col(k)
                    if f == feature and g == group and normalize_test_name(t) == tn:
                        if m == "pass":
                            is_significant = str(v).strip().upper() in (
                                "OK",
                                "TRUE",
                                "1",
                            )
                            rec[f"{tn} pass"] = "NOT OK" if is_significant else "OK"
                        elif m == "p-value":
                            rec[f"{tn} p-value"] = v

            result_rows.append(rec)

        roles = {}
        if result_rows:
            for c in result_rows[0]:
                if c in ("feature", "group"):
                    roles[c] = InfoRole()
                else:
                    roles[c] = StatisticRole()

        self.best_split_statistics = SmallDataset.from_dict(result_rows, roles=roles)

    def _extract_experiments(self, experiment_data: ExperimentData):
        id_ = experiment_data.get_one_id(
            "ParamsExperiment", ExperimentDataEnum.analysis_tables
        )
        raw_table = experiment_data.analysis_tables[id_]
        pdf = raw_table.raw_data
        result_rows = []
        for row_idx in range(len(pdf)):
            row: dict = {}
            for col in pdf.columns:
                val = pdf.iloc[row_idx].get(col)
                col_str = str(col)
                if NAME_BORDER_SYMBOL in col_str:
                    continue
                if ID_SPLIT_SYMBOL in col_str:
                    parts = col_str.split(ID_SPLIT_SYMBOL)
                    norm_parts = [normalize_test_name(p) for p in parts]
                    new_col = " ".join(norm_parts)
                else:
                    new_col = col_str

                for raw_name, norm_name in TEST_NAME_NORMALIZATION.items():
                    if raw_name != norm_name:
                        new_col = new_col.replace(raw_name, norm_name)

                new_col = new_col.replace(" all", "")

                if "pass" in new_col and not new_col.startswith("mean "):
                    if val is not None and not (
                        isinstance(val, float) and math.isnan(val)
                    ):
                        if isinstance(val, str):
                            val = val.strip().lower() in ("true", "1", "ok")
                        else:
                            val = bool(val)

                row[new_col] = val
            result_rows.append(row)

        if result_rows:
            self.experiments = SmallDataset.from_dict(result_rows, roles={})
        else:
            self.experiments = SmallDataset.from_dict(
                [{"feature": [], "group": []}], roles={}
            )
        self.experiments = self._replace_splitters(self.experiments, RenameEnum.columns)

    @staticmethod
    def _add_dry_score(experiment_data: ExperimentData, aa_score: Dataset) -> Dataset:
        dry_score_analyser_ids = experiment_data.get_ids(
            AADryTestAnalyzer, ExperimentDataEnum.analysis_tables
        )[AADryTestAnalyzer.__name__][ExperimentDataEnum.analysis_tables.value]

        if len(dry_score_analyser_ids) == 0:
            return aa_score

        dry_score_analyser_ids = dry_score_analyser_ids[0]
        table = aa_score.merge(
            experiment_data.analysis_tables[dry_score_analyser_ids],
            right_index=True,
            left_index=True,
            how="left",
        )
        new_pass_col = (table["pass_x"].fillna(1) + table["pass_y"].fillna(1)).rename(
            {"pass_x": "pass"}
        )
        table.add_column(new_pass_col)
        table = table.drop(columns=["pass_x", "pass_y"])
        return table

    def _extract_aa_score(self, experiment_data: ExperimentData):
        def get_analyzer_id(key: str):
            target_id = [i for i in aa_score_analyser_ids if i.endswith(key)]
            if len(target_id):
                return target_id[0]
            raise ValueError("Result of AAScoreAnalyzer does not found.")

        aa_score_analyser_ids = experiment_data.get_ids(
            AAScoreAnalyzer, ExperimentDataEnum.analysis_tables
        )[AAScoreAnalyzer.__name__][ExperimentDataEnum.analysis_tables.value]

        self.aa_score = experiment_data.analysis_tables[get_analyzer_id("aa score")]
        self.aa_score = self._add_dry_score(experiment_data, self.aa_score)
        self.aa_score = self._replace_splitters(self.aa_score, RenameEnum.index)

        self.best_split_statistics = experiment_data.analysis_tables[
            get_analyzer_id("best split statistics")
        ]

        rename_map = {}
        for col in self.best_split_statistics.columns:
            for raw, norm in TEST_NAME_NORMALIZATION.items():
                if col.startswith(raw + " ") and raw != norm:
                    rename_map[col] = col.replace(raw, norm, 1)
                    break
        if rename_map:
            try:
                self.best_split_statistics.raw_data = (
                    self.best_split_statistics.raw_data.rename(columns=rename_map)
                )
                self.best_split_statistics._roles = {
                    rename_map.get(c, c): r
                    for c, r in self.best_split_statistics._roles.items()
                }
            except Exception:
                # best-effort rename: keep the original column names on failure
                pass

    def extract(self, experiment_data: ExperimentData):
        super().extract(experiment_data)
        self._extract_experiments(experiment_data)
        self._extract_aa_score(experiment_data)
        self._extract_best_split_statistics(experiment_data)
    
    def _summary_stats(self) -> dict[str, Any]:
        """Collect high-level AA statistics for the summary block."""
        stats: dict[str, Any] = {}

        # Iterations count
        if hasattr(self, "experiments") and self.experiments is not None:
            stats["iterations"] = len(self.experiments)

        # AA score (mean pass-rate weight)
        if hasattr(self, "aa_score") and self.aa_score is not None:
            try:
                scores = [
                    r.get("score", 0) for r in self.aa_score.to_records()
                ]
                passes = [
                    r.get("pass", False) for r in self.aa_score.to_records()
                ]
                stats["mean_score"] = sum(scores) / max(len(scores), 1)
                stats["passed"] = sum(1 for p in passes if p)
                stats["total"] = len(passes)
            except Exception:
                pass

        # Best split id
        if hasattr(self, "best_split_statistics") and self.best_split_statistics is not None:
            try:
                recs = self.best_split_statistics.to_records()
                if recs and "splitter_id" in recs[0]:
                    stats["best_split"] = recs[0]["splitter_id"]
            except Exception:
                pass

        return stats

    def __repr__(self) -> str:
        s = self._summary_stats()
        lines = ["AAOutput"]

        if "iterations" in s:
            lines.append(f"  Iterations : {s['iterations']}")
        if "mean_score" in s:
            lines.append(f"  AA Score   : {s['mean_score']:.3f}")
        if "passed" in s and "total" in s:
            lines.append(f"  Passed     : {s['passed']}/{s['total']}")
        if "best_split" in s:
            lines.append(f"  Best Split : {s['best_split']}")

        lines.append("")
        lines.append(repr(self._report()))
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        s = self._summary_stats()
        parts: list[str] = [
            "<div style='font-family:monospace; display:inline-block;'>",
            "<b style='font-size:14px;'>AAOutput</b>",
            "<table style='border-collapse:collapse; margin:6px 0; font-size:12px;'>",
        ]
        if "iterations" in s:
            parts.append(
                f"<tr><td style='padding:2px 8px; color:#555;'>Iterations</td>"
                f"<td style='padding:2px 8px; font-weight:bold;'>{s['iterations']}</td></tr>"
            )
        if "mean_score" in s:
            color = "#2e7d32" if s["mean_score"] >= 0.8 else "#f57f17"
            parts.append(
                f"<tr><td style='padding:2px 8px; color:#555;'>AA Score</td>"
                f"<td style='padding:2px 8px; font-weight:bold; color:{color};'>"
                f"{s['mean_score']:.3f}</td></tr>"
            )
        if "passed" in s and "total" in s:
            color = "#2e7d32" if s["passed"] == s["total"] else "#c62828"
            parts.append(
                f"<tr><td style='padding:2px 8px; color:#555;'>Passed</td>"
                f"<td style='padding:2px 8px; font-weight:bold; color:{color};'>"
                f"{s['passed']}/{s['total']}</td></tr>"
            )
        if "best_split" in s:
            parts.append(
                f"<tr><td style='padding:2px 8px; color:#555;'>Best Split</td>"
                f"<td style='padding:2px 8px;'><code>{html.escape(str(s['best_split']))}</code></td></tr>"
            )
        parts.append("</table>")
        parts.append(self._report()._repr_html_())
        parts.append("</div>")
        return "".join(parts)
