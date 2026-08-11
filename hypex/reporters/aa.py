from __future__ import annotations
from typing import Any, ClassVar
import warnings

from ..comparators import (
    GroupChi2Test, GroupDifference, GroupKSTest, GroupTTest,
    StatsTTest, StatsKSTest, StatsChi2Test, BaseComparator
)
from ..comparators.adaptive_hypothesis_testing import TTest, KSTest, Chi2Test, UTest

from ..dataset import Dataset, ExperimentData, InfoRole, StatisticRole
from ..dataset.dataset import SmallDataset
from ..splitters import AASplitter, AASplitterWithStratification
from ..utils import ID_SPLIT_SYMBOL, ExperimentDataEnum, NotFoundInExperimentDataError
from .abstract import (
    DictReporter, DatasetReporter, Reporter, 
    extract_group_difference, extract_tests, extract_analyzer_data
)

class AATestReporter(DatasetReporter):
    """Reporter for A/A test results.

    Extracts group differences, statistical test outcomes, and analyzer metadata,
    formatting them into a structured dataset or dictionary.
    """
    tests: ClassVar[list[type[BaseComparator]]] = [
        GroupTTest, GroupKSTest, GroupChi2Test,
        StatsTTest, StatsKSTest, StatsChi2Test,
        TTest, KSTest, Chi2Test, UTest,         
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
    """Reporter for A/A test pass/fail results.

    Produces the legacy resume format:
        feature | group | TTest aa test | KSTest aa test | TTest best split |
        KSTest best split | result | control mean | test mean | difference | difference %
    """

    def report(self, data: ExperimentData) -> Dataset:
        analyser_ids = data.get_ids(
            "AAScoreAnalyzer", ExperimentDataEnum.analysis_tables
        )
        analyser_tables = {
            id_[id_.rfind(ID_SPLIT_SYMBOL) + 1 :]: data.analysis_tables[id_]
            for id_ in analyser_ids["AAScoreAnalyzer"][
                ExperimentDataEnum.analysis_tables.value
            ]
        }

        if not analyser_tables.get("aa score") or analyser_tables["aa score"].is_empty():
            return None

        aa_score = analyser_tables["aa score"]
        best_split_stats = analyser_tables.get("best split statistics")

        if best_split_stats is None or best_split_stats.is_empty():
            return SmallDataset.create_empty()

        # --- collect test display names from aa_score index ---
        # index labels look like "pre_spends TTest test_1"
        test_names_ordered: list[str] = []
        seen: set[str] = set()
        for idx_label in aa_score.index:
            parts = str(idx_label).split()
            # parts: [feature, TestName, group] or [TestName, group]
            if len(parts) >= 2:
                tn = parts[-2] if len(parts) >= 3 else parts[0]
                if tn not in seen:
                    seen.add(tn)
                    test_names_ordered.append(tn)

        # --- build rows ---
        records = best_split_stats.to_records()
        result_records: list[dict] = []

        for row in records:
            feature = row.get("feature", "")
            group = row.get("group", "")
            rec: dict = {"feature": feature, "group": group}

            # aa test columns (from aa_score)
            for tn in test_names_ordered:
                idx_key = f"{feature} {tn} {group}"
                try:
                    pass_val = aa_score.loc[idx_key, "pass"]
                    rec[f"{tn} aa test"] = "OK" if pass_val else "NOT OK"
                except Exception:
                    rec[f"{tn} aa test"] = None

            # best split columns
            for tn in test_names_ordered:
                pass_col = f"{tn} pass"
                # search in row keys (may have normalized or raw name)
                val = None
                for k, v in row.items():
                    if k.lower().endswith("pass") and normalize_test_name(
                        k[: k.lower().rfind("pass")].strip()
                    ) == tn:
                        val = v
                        break
                if val is not None:
                    rec[f"{tn} best split"] = (
                        "OK"
                        if str(val).strip().upper() in ("OK", "TRUE", "1")
                        else "NOT OK"
                    )
                else:
                    rec[f"{tn} best split"] = None

            # result column
            all_ok = all(
                rec.get(f"{tn} best split") != "NOT OK"
                for tn in test_names_ordered
                if rec.get(f"{tn} best split") is not None
            )
            rec["result"] = "OK" if all_ok else "NOT OK"

            # numeric columns
            for nc in ("control mean", "test mean", "difference", "difference %"):
                rec[nc] = row.get(nc)

            result_records.append(rec)

        roles = {
            "feature": InfoRole(),
            "group": InfoRole(),
            "result": StatisticRole(),
        }
        for c in result_records[0]:
            if c not in roles:
                roles[c] = StatisticRole()

        return SmallDataset.from_dict(result_records, roles=roles)


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
        best_split_id = next((c for c in data.additional_fields.columns if c.endswith("best")), None)
        if best_split_id is None:
            return data.ds
        markers = data.additional_fields.select([best_split_id])
        markers = markers.rename({best_split_id: "split"})
        return data.ds.merge(markers, left_index=True, right_index=True)