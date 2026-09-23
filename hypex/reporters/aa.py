from __future__ import annotations

import warnings
from typing import Any, ClassVar

from ..comparators import (
    BaseComparator,
    GroupChi2Test,
    GroupKSTest,
    GroupTTest,
    StatsChi2Test,
    StatsKSTest,
    StatsTTest,
)
from ..dataset import Dataset, ExperimentData, InfoRole, StatisticRole
from ..dataset.dataset import SmallDataset
from ..splitters import AASplitter, AASplitterWithStratification
from ..utils import ID_SPLIT_SYMBOL, ExperimentDataEnum, NotFoundInExperimentDataError
from ..utils.constants import NAME_BORDER_SYMBOL
from ..utils.naming import _parse_metric_col, normalize_test_name
from .abstract import (
    DatasetReporter,
    DictReporter,
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
        GroupTTest, GroupKSTest, GroupChi2Test,
        StatsTTest, StatsKSTest, StatsChi2Test,
    ]

    def __init__(self, 
                 dict_reporter: DictReporter | None = None, 
                 output_format: str = "dataset"):
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

        Args:
            data: The experiment data container.

        Returns:
            The ID of the ``AASplitter`` or ``AASplitterWithStratification`` instance, 
            or ``None`` if no splitter is found.
        """
        for c in [AASplitter, AASplitterWithStratification]:
            try:
                return data.get_one_id(c, ExperimentDataEnum.additional_fields)
            except NotFoundInExperimentDataError:
                pass

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

class OneAADictReporter(AATestReporter):
    """Legacy reporter wrapper for dictionary output.

    Deprecated: Use ``AATestReporter(output_format='dict')`` instead.
    """
    def __init__(self, front: bool = True):
        """Initialize the legacy dictionary reporter.

        Args:
            front: If ``True``, formats keys for front-end display.
            Defaults to ``True``.
        """
        super().__init__(dict_reporter=DictReporter(front=front), output_format="dict")
        warnings.warn("OneAADictReporter is deprecated. Use AATestReporter(output_format='dict')", 
                      DeprecationWarning, 
                      stacklevel=2)

    @staticmethod
    def convert_flat_dataset(data: dict[str, Any]) -> SmallDataset:
        """Convert a flat dictionary representation into a ``SmallDataset``.

        Args:
            data: The flat dictionary to convert.

        Returns:
            A ``SmallDataset`` instance containing the structured data.
        """
        return AATestReporter.convert_to_dataset(data)

class AADatasetReporter(AATestReporter):
    """Legacy reporter wrapper for dataset output.

    Deprecated: Use ``AATestReporter()`` instead.
    """
    def __init__(self):
        """Initialize the legacy dataset reporter."""
        super().__init__(dict_reporter=DictReporter(), output_format="dataset")
        warnings.warn("AADatasetReporter is deprecated. Use AATestReporter()", 
                      DeprecationWarning, 
                      stacklevel=2)


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
            id_[id_.rfind(ID_SPLIT_SYMBOL) + 1:]: data.analysis_tables[id_]
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
        order_map = {"TTest": 0, "KSTest": 1, "Chi2Test": 2, "ZTest": 3}
        names = dict.fromkeys(
            str(idx).split()[-2] if len(str(idx).split()) >= 3
            else str(idx).split()[0]
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
            if f == feature and normalize_test_name(t) == tn and m == "pass" and g == group:
                return "NOT OK" if str(v).strip().upper() in ("OK", "TRUE", "1") else "OK"
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
