from __future__ import annotations

import os
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np

from ..comparators import (
    BaseComparator,
    GroupChi2Test,
    GroupKSTest,
    GroupTTest,
    StatsChi2Test,
    StatsKSTest,
    StatsTTest,
    StatsUTest,
)
from ..dataset import Dataset, ExperimentData, InfoRole, StatisticRole
from ..dataset.dataset import SmallDataset
from ..splitters import AASplitter, AASplitterWithStratification
from ..utils import ID_SPLIT_SYMBOL, ExperimentDataEnum
from ..utils.constants import NAME_BORDER_SYMBOL
from ..utils.naming import _parse_metric_col, normalize_test_name
from .abstract import (
    DatasetReporter,
    DictReporter,
    PlotReporter,
    Reporter,
    extract_analyzer_data,
    extract_group_difference,
    extract_tests,
)


class AATestReporter(DatasetReporter):
    """Reporter for A/A test results.

    Extracts group differences, statistical test outcomes, and analyzer metadata,
    formatting them into a structured dataset or dictionary.
    """

    tests: ClassVar[list[type[BaseComparator]]] = [
        GroupTTest,
        GroupKSTest,
        GroupChi2Test,
        StatsTTest,
        StatsKSTest,
        StatsChi2Test,
        StatsUTest,
    ]

    def __init__(
        self, dict_reporter: DictReporter | None = None, output_format: str = "dataset"
    ):
        """Initialize the A/A test reporter.

        Args:
            dict_reporter: A ``DictReporter`` instance to handle dictionary formatting.
                If ``None``, a default ``DictReporter`` is created.
            output_format: The desired output format. Must be ``'dict'`` or ``'dataset'``.
        """
        if dict_reporter is None:
            dict_reporter = DictReporter()
        super().__init__(dict_reporter, output_format)

    @staticmethod
    def get_splitter_id(data: ExperimentData) -> str | None:
        """Retrieve the identifier of the splitter used in the experiment.

        Searches for ``AASplitterWithStratification`` first (the more
        specific class) to avoid a guaranteed miss-and-retry cycle when
        stratification is enabled.

        Args:
            data: The experiment data container.

        Returns:
            The splitter ID string, or ``None`` if no splitter is found.
        """
        for c in [AASplitterWithStratification, AASplitter]:
            ids = data.get_ids(c, ExperimentDataEnum.additional_fields)
            found = ids.get(c.__name__, {}).get(
                ExperimentDataEnum.additional_fields.value,
                [],
            )
            if found:
                return found[0]
        return None

    def _build_dict_report(self, data: ExperimentData) -> dict[str, Any]:
        """Construct a dictionary report containing A/A test metrics.

        Args:
            data: The experiment data container.

        Returns:
            A dictionary with splitter ID, group differences, test results, and analyzer data.
        """
        result = {"splitter_id": self.get_splitter_id(data)}
        front_flag = self.dict_reporter.front
        result.update(extract_group_difference(data, front_flag))
        result.update(extract_tests(data, self.tests, front_flag))
        result.update(extract_analyzer_data(data, "OneAAStatAnalyzer"))

        return result

    def report(self, data: ExperimentData) -> dict[str, Any] | Dataset:
        """Generate the final A/A test report.

        Args:
            data: The experiment data container.

        Returns:
            The report as a dictionary or ``Dataset``, depending on the configured ``output_format``.
        """
        prev = self.dict_reporter.front
        self.dict_reporter.front = False
        try:
            dict_result = self._build_dict_report(data)
            if self.output_format == "dict":
                return dict_result
            return self.convert_to_dataset(dict_result)
        finally:
            self.dict_reporter.front = prev


class AAPassedReporter(Reporter):
    def report(self, data: ExperimentData) -> Dataset:
        aa_score, best_split = self._collect_tables(data)
        if aa_score is None or best_split is None:
            return SmallDataset.create_empty()

        test_names = self._ordered_test_names(aa_score)

        records = []
        for row in best_split.to_records():
            for feature, group in self._feature_groups(row):
                rec = self._build_record(row, feature, group, test_names, aa_score)
                records.append(rec)

        return self._to_dataset(records)

    # ── collecting tables ────────────────────────────────────────────────

    @staticmethod
    def _collect_tables(data: ExperimentData):
        ids = data.get_ids("AAScoreAnalyzer", ExperimentDataEnum.analysis_tables)
        tables = {
            id_[id_.rfind(ID_SPLIT_SYMBOL) + 1 :]: data.analysis_tables[id_]
            for id_ in ids.get("AAScoreAnalyzer", {}).get("analysis_tables", [])
        }
        aa_score = tables.get("aa score")
        best_split = tables.get("best split statistics")

        if aa_score is None or aa_score.is_empty():
            return None, None
        if best_split is None or best_split.is_empty():
            return None, None

        return aa_score, best_split

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _ordered_test_names(aa_score: Dataset) -> list[str]:
        order_map = {"TTest": 0, "KSTest": 1, "Chi2Test": 2, "UTest": 3, "ZTest": 4}
        names = dict.fromkeys(
            str(idx).split()[-2] if len(str(idx).split()) >= 3 else str(idx).split()[0]
            for idx in aa_score.index
        )
        return sorted(names, key=lambda t: order_map.get(t, 99))

    @staticmethod
    def _feature_groups(row: dict) -> list[tuple[str, str]]:
        groups = set()
        for k in row:
            if NAME_BORDER_SYMBOL in k:
                continue
            f, _, _, g = _parse_metric_col(k)
            if f and f != "mean":
                groups.add((f, g))
        return sorted(groups)

    # ── building record ───────────────────────────────────────────────────

    def _build_record(self, row, feature, group, test_names, aa_score) -> dict:
        rec: dict = {"feature": feature, "group": group}

        for tn in test_names:
            idx_key = f"{feature} {tn} {group}".strip()

            rec[f"{tn} aa score"] = self._aa_pass(aa_score, idx_key)

            rec[f"{tn} best split"] = self._best_split_pass(row, feature, tn, group)

        failed = any(
            rec.get(f"{tn} {sfx}") == "NOT OK"
            for tn in test_names
            for sfx in ("aa score", "best split")
        )
        rec["result"] = "NOT OK" if failed else "OK"

        for m in ("control mean", "test mean", "difference", "difference %"):
            rec[m] = self._metric(row, feature, "GroupDifference", m, group)

        return rec

    # ── value extractors ───────────────────────────────────────────────────

    @staticmethod
    def _aa_pass(aa_score: Dataset, idx_key: str):
        try:
            v = aa_score.get_values(row=idx_key, column="pass")
            return "OK" if v else "NOT OK"
        except Exception:
            return None

    @staticmethod
    def _best_split_pass(row: dict, feature: str, tn: str, group: str):
        for k, v in row.items():
            f, t, m, g = _parse_metric_col(k)
            if (
                f == feature
                and normalize_test_name(t) == tn
                and m == "pass"
                and g == group
            ):
                return (
                    "NOT OK" if str(v).strip().upper() in ("OK", "TRUE", "1") else "OK"
                )
        return None

    @staticmethod
    def _metric(row: dict, feature: str, test: str, metric: str, group: str):
        for k, v in row.items():
            f, t, m, g = _parse_metric_col(k)
            if f == feature and t == test and m == metric and g == group:
                return v
        return None

    # ── dataset assembly ───────────────────────────────────────────────────

    @staticmethod
    def _to_dataset(records: list[dict]) -> SmallDataset:
        roles: dict = {
            "feature": InfoRole(),
            "group": InfoRole(),
            "result": StatisticRole(),
        }
        if records:
            for c in records[0]:
                if c not in roles:
                    roles[c] = StatisticRole()
        return SmallDataset.from_dict(records, roles=roles)


class AABestSplitReporter(Reporter):
    """Reporter that attaches best split markers to the dataset.

    Identifies the optimal data split and merges its identifier back into
    the primary dataset for downstream analysis.
    """

    def report(self, data: ExperimentData) -> Dataset:
        """Merge the best split identifier into the main dataset.

        Args:
            data: The experiment data container.

        Returns:
            The original dataset merged with a 'split' column indicating the
            best split configuration.
        """
        best_split_id = next(
            (c for c in data.additional_fields.columns if c.endswith("best")),
            None,
        )
        if best_split_id is None:
            return data.ds

        markers = data.additional_fields.select([best_split_id])
        markers = markers.rename({best_split_id: "split"})
        result = data.ds.merge(markers, left_index=True, right_index=True)

        if best_split_id in result.columns:
            result = result.drop(columns=[best_split_id])
        return result

class AADryTestPlotReporter(PlotReporter):

    def __init__(
            self,
            figure_dpi,
            save_fig=False,
            save_dir: str = ".",
            alpha: float = 0.05
        ):
        super().__init__(figure_dpi, save_fig)
        self.save_dir = save_dir
        self._saved_paths: dict[str, str] = {}
        self.alpha = alpha

    # ── create figure ────────────────────────────────────────────────

    def _create_figure(self, data: ExperimentData) -> dict[Any, plt.Figure]:
        result = extract_analyzer_data(
            data,
            "AADryTestAnalyzer",
            orient="dataset",
            key="raw pvals",
        )
        if not result:
            return {}

        table = result["result"]
        if table.is_empty():
            return {}

        n_cols = len(table.columns)
        if n_cols == 0:
            return {}

        fig, axs = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
        fig.suptitle(
            "AA-test: diagnostic with zero effect",
            fontsize=15, fontweight="bold",
        )

        if n_cols == 1:
            axs = [axs]

        for idx, col in enumerate(table.columns):
            ax = axs[idx]
            p_vals = table[col].raw_data

            ax.hist(
                p_vals, bins=30, density=True,
                alpha=0.7, color="steelblue", edgecolor="black",
            )
            ax.axhline(y=1.0, color="red", linestyle="--",
                       linewidth=2, label="Uniform(0,1)")
            ax.axvline(x=self.alpha, color="green", linestyle=":",
                       linewidth=2, label=f"α = {self.alpha}")
            ax.set_title(f"{col}: p-values distribution")
            ax.set_xlabel("p-value")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)

            fpr = float(np.mean(p_vals < self.alpha))
            ax.text(
                0.5, 0.95,
                f"FPR = {fpr:.1%}\n(expected: {self.alpha:.0%})",
                transform=ax.transAxes, ha="center", va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                fontsize=11,
            )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return {"aa_dry_test": fig}

    # ── save figure ────────────────────────────────────────────────

    def _save_figure(self, figures: dict[str, plt.Figure]):
        os.makedirs(self.save_dir, exist_ok=True)
        paths = {}
        for name, fig in figures.items():
            path = os.path.join(self.save_dir, f"{name}.png")
            fig.savefig(path, dpi=self.figure_dpi, bbox_inches="tight")
            paths[name] = path
        self._saved_paths = paths
        return paths

    @staticmethod
    def _show_figure(figure: dict[str, plt.Figure]) -> None:
        plt.show(block=False)

    # ── Interactive display (for scripts outside Jupyter) ───────────────

    @staticmethod
    def _show_figure(figures: dict[str, plt.Figure]) -> None:
        plt.show(block=False)

    # ── report ────────────────────────────────────────────────────────

    def report(self, data: ExperimentData) -> dict[str, plt.Figure]:
        fig_dict = self._create_figure(data)
        if not fig_dict:
            return {}
        if self.save_fig:
            self._save_figure(fig_dict)
        return fig_dict

