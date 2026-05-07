from __future__ import annotations
from typing import Any, ClassVar
import warnings

from ..comparators import GroupChi2Test, GroupDifference, GroupKSTest, GroupTTest, StatsTTest
from ..dataset import Dataset, ExperimentData, InfoRole, StatisticRole
from ..dataset.dataset import SmallDataset
from ..splitters import AASplitter, AASplitterWithStratification
from ..utils import ID_SPLIT_SYMBOL, ExperimentDataEnum, NotFoundInExperimentDataError
from .abstract import (
    DictReporter, DatasetReporter, Reporter, 
    extract_group_difference, extract_tests, extract_analyzer_data
)

class AATestReporter(DatasetReporter):
    tests: ClassVar[list] = [GroupTTest, GroupKSTest, GroupChi2Test, StatsTTest]

    def __init__(
        self, 
        dict_reporter: DictReporter | None = None, 
        output_format: str = "dataset"
    ):
        if dict_reporter is None:
            dict_reporter = DictReporter()
        super().__init__(dict_reporter, output_format)

    @staticmethod
    def get_splitter_id(data: ExperimentData):
        for c in [AASplitter, AASplitterWithStratification]:
            try:
                return data.get_one_id(c, ExperimentDataEnum.additional_fields)
            except NotFoundInExperimentDataError:
                pass

    def _build_dict_report(self, data: ExperimentData) -> dict[str, Any]:
        result = {"splitter_id": self.get_splitter_id(data)}
        front_flag = self.dict_reporter.front
        result.update(extract_group_difference(data, front_flag))
        result.update(extract_tests(data, self.tests, front_flag))
        result.update(extract_analyzer_data(data, "OneAAStatAnalyzer"))
        return result

    def report(self, data: ExperimentData) -> dict | Dataset:
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
    """Legacy wrapper. Preserves old constructor signature."""
    def __init__(self, front: bool = True):
        super().__init__(dict_reporter=DictReporter(front=front), output_format="dict")
        warnings.warn("OneAADictReporter is deprecated. Use AATestReporter(output_format='dict')", DeprecationWarning, stacklevel=2)

    @staticmethod
    def convert_flat_dataset(data: dict) -> SmallDataset:
        return AATestReporter.convert_to_dataset(data)

class AADatasetReporter(AATestReporter):
    """Legacy wrapper."""
    def __init__(self):
        super().__init__(dict_reporter=DictReporter(), output_format="dataset")
        warnings.warn("AADatasetReporter is deprecated. Use AATestReporter()", DeprecationWarning, stacklevel=2)


class AAPassedReporter(Reporter):
    def report(self, data: ExperimentData) -> Dataset:
        analyser_ids = data.get_ids("AAScoreAnalyzer", ExperimentDataEnum.analysis_tables)
        analyser_tables = {
            id_[id_.rfind(ID_SPLIT_SYMBOL) + 1:]: data.analysis_tables[id_]
            for id_ in analyser_ids["AAScoreAnalyzer"][ExperimentDataEnum.analysis_tables.value]
        }
        if not analyser_tables.get("aa score") or analyser_tables["aa score"].is_empty():
            return None

        score_table = self._reformat_bool(analyser_tables["aa score"])
        best_split_table = self._reformat_bool_split(analyser_tables["best split statistics"])

        if best_split_table.is_empty():
            return SmallDataset.create_empty()

        records = []
        for i in range(len(best_split_table)):
            rec = {"feature": str(best_split_table.get_values(i, "feature")),
                   "group": str(best_split_table.get_values(i, "group"))}
            passed = False
            for col in best_split_table.columns:
                if col in ("feature", "group"): continue
                val = best_split_table.get_values(i, col)
                rec[col] = val
                if val in (True, 1, "True", 1.0):
                    passed = True
            rec["result"] = "OK" if passed else "NOT OK"
            records.append(rec)

        roles = {"feature": InfoRole(), "group": InfoRole(), "result": StatisticRole()}
        for col in best_split_table.columns:
            if col not in ("feature", "group"):
                roles[col] = best_split_table.roles.get(col, StatisticRole())

        result = SmallDataset.from_dict(records, roles=roles)
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
                try: result.data[col] = result.data[col].astype(float).round(6)
                except Exception: pass
        return result

    @staticmethod
    def _reformat_bool(table: Dataset) -> Dataset:
        if table.is_empty() or "pass" not in table.columns:
            return SmallDataset.create_empty()
        
        pass_dict = table.data.iloc[0]["pass"]
        
        if not isinstance(pass_dict, dict) or not pass_dict:
            return SmallDataset.create_empty()
        return SmallDataset.from_dict(pass_dict, roles={k: InfoRole() for k in pass_dict})

    @staticmethod
    def _reformat_bool_split(table: Dataset) -> Dataset:
        pass_cols = [c for c in table.columns if c.endswith("pass")]
        if not pass_cols or table.is_empty():
            return SmallDataset.create_empty()
            
        rows = []
        for i in range(len(table)):
            row_dict = {}
            if "feature" in table.columns:
                row_dict["feature"] = table.data.iloc[i]["feature"]
            if "group" in table.columns:
                row_dict["group"] = table.data.iloc[i]["group"]
                
            for col in pass_cols:
                val = table.data.iloc[i][col]
                row_dict[col[:col.rfind("pass")-1].strip()] = bool(val)
            rows.append(row_dict)
            
        return SmallDataset.from_dict(rows[0] if rows else {}, roles={c: InfoRole() for c in rows[0]})

class AABestSplitReporter(Reporter):
    def report(self, data: ExperimentData):
        best_split_id = next((c for c in data.additional_fields.columns if c.endswith("best")), None)
        if best_split_id is None:
            return data.ds
        markers = data.additional_fields.select([best_split_id])
        markers = markers.rename({best_split_id: "split"})
        return data.ds.merge(markers, left_index=True, right_index=True)