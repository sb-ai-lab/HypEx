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

    Aggregates test pass statuses, reformats boolean indicators, 
    and merges statistical differences (mean, p-value, etc.) into a final summary table.
    """
    def report(self, data: ExperimentData) -> Dataset:
        """Generate a report summarizing A/A test pass/fail outcomes.

        Args:
            data: The experiment data container.

        Returns:
            A ``Dataset`` containing pass/fail statuses, statistical metrics,
            and an overall result flag, or ``None`` if the required analyzer data is missing or empty.
        """
        analyser_ids = data.get_ids("AAScoreAnalyzer", ExperimentDataEnum.analysis_tables)
        analyser_tables = {
            id_[id_.rfind(ID_SPLIT_SYMBOL) + 1:]: data.analysis_tables[id_]
            for id_ in analyser_ids["AAScoreAnalyzer"][ExperimentDataEnum.analysis_tables.value]
        }
        if not analyser_tables.get("aa score") or analyser_tables["aa score"].is_empty():
            return None

        best_split_table = self._reformat_bool_split(analyser_tables["best split statistics"])

        if best_split_table.is_empty():
            return SmallDataset.create_empty()

        records = best_split_table.to_records()
        result_records = []
        
        for row in records:
            rec = {
                "feature": str(row.get("feature", "")),
                "group": str(row.get("group", ""))
            }
            passed = False
            for col in best_split_table.columns:
                if col in ("feature", "group"): 
                    continue
                val = row.get(col)
                rec[col] = val
                if val in (True, 1, "True", 1.0, "OK"):
                    passed = True
            rec["result"] = "OK" if passed else "NOT OK"
            result_records.append(rec)

        roles = {"feature": InfoRole(), "group": InfoRole(), "result": StatisticRole()}
        for col in best_split_table.columns:
            if col not in ("feature", "group"):
                roles[col] = best_split_table.roles.get(col, StatisticRole())

        result = SmallDataset.from_dict(result_records, roles=roles)
        
        diff_source = analyser_tables.get("best split statistics")
        if diff_source and not diff_source.is_empty():
            stats_cols = ["feature", "group", "control mean", "test mean", "difference", "difference %"]
            available = [c for c in stats_cols if c in diff_source.columns]
            if available:
                differences = diff_source.select(available)
                try:
                    result = result.merge(differences, on=["feature", "group"], how="left")
                except Exception:
                    pass

        numeric_cols = ["control mean", "test mean", "difference", "difference %"]
        for col in numeric_cols:
            if col in result.columns:
                try: 
                    result.data[col] = result.data[col].astype(float).round(6)
                except Exception: 
                    pass
        return result

    @staticmethod
    def _reformat_bool(table: SmallDataset) -> SmallDataset:
        """Extract and reformat pass/fail statuses from a raw results table.

        Args:
            table: The raw dataset containing a 'pass' column with nested dictionaries.

        Returns:
            A ``SmallDataset`` with reformatted pass/fail statuses, 
            or an empty ``SmallDataset`` if the input is invalid or empty.
        """
        if table.is_empty() or "pass" not in table.columns:
            return SmallDataset.create_empty()
        
        records = table.to_records()
        pass_dict = records[0].get("pass") if records else None
        
        if not isinstance(pass_dict, dict) or not pass_dict:
            return SmallDataset.create_empty()
        return SmallDataset.from_dict(pass_dict, roles={k: InfoRole() for k in pass_dict})

    @staticmethod
    def _reformat_bool_split(table: SmallDataset) -> SmallDataset:
        """Extract and reformat split-specific pass/fail statuses.

        Args:
            table: The raw dataset containing columns ending with 'pass'.

        Returns:
            A ``SmallDataset`` with cleaned boolean pass/fail values per split,
            or an empty ``SmallDataset`` if no relevant columns are found.
        """
        pass_cols = [c for c in table.columns if c.endswith("pass")]
        if not pass_cols or table.is_empty():
            return SmallDataset.create_empty()
            
        records = table.to_records()
        rows = []
        for row in records:
            row_dict = {}
            if "feature" in row:
                row_dict["feature"] = row["feature"]
            if "group" in row:
                row_dict["group"] = row["group"]
                
            for col in pass_cols:
                val = row.get(col)

                clean_col = col[:col.rfind("pass")].strip()
                row_dict[clean_col] = bool(val) if val is not None else False
            rows.append(row_dict)
            
        if not rows:
            return SmallDataset.create_empty()
            
        return SmallDataset.from_dict(rows, roles={c: InfoRole() for c in rows[0]})

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