from __future__ import annotations

from typing import Any, ClassVar

from ..comparators import GroupChi2Test, GroupDifference, GroupSizes, GroupKSTest, GroupTTest, StatsTTest
from ..dataset import Dataset, ExperimentData, InfoRole, StatisticRole
from ..dataset.dataset import SmallDataset
from ..splitters import AASplitter, AASplitterWithStratification
from ..utils import ID_SPLIT_SYMBOL, ExperimentDataEnum, NotFoundInExperimentDataError, BackendsEnum
from .abstract import Reporter, TestDictReporter
from ..utils import RoleColumnError


class OneAADictReporter(TestDictReporter):
    tests: ClassVar[list] = [GroupTTest, GroupKSTest, GroupChi2Test, StatsTTest]

    @staticmethod
    def convert_flat_dataset(data: dict) -> SmallDataset:
        struct_dict = OneAADictReporter._get_struct_dict(data)
        return OneAADictReporter._convert_struct_dict_to_dataset(struct_dict)

    @staticmethod
    def get_splitter_id(data: ExperimentData):
        for c in [AASplitter, AASplitterWithStratification]:
            try:
                return data.get_one_id(c, ExperimentDataEnum.additional_fields)
            except NotFoundInExperimentDataError:
                pass  # The splitting was done by another class

    def extract_group_difference(self, data: ExperimentData) -> dict[str, Any]:
        group_difference_ids = data.get_ids(GroupDifference)[GroupDifference.__name__][
            ExperimentDataEnum.analysis_tables.value
        ]
        return self._extract_from_comparators(data, group_difference_ids)

    def extract_group_sizes(self, data: ExperimentData) -> dict[str, Any]:
        group_sizes_id = data.get_one_id(GroupSizes, ExperimentDataEnum.analysis_tables)
        return self._extract_from_comparators(data, [group_sizes_id])

    def extract_analyzer_data(self, data: ExperimentData) -> dict[str, Any]:
        analyzer_id = data.get_one_id(
            "OneAAStatAnalyzer", ExperimentDataEnum.analysis_tables
        )
        return self.extract_from_one_row_dataset(data.analysis_tables[analyzer_id])

    def extract_data_from_analysis_tables(self, data: ExperimentData) -> dict[str, Any]:
        result = {}
        result.update(self.extract_group_difference(data))
        # result.update(self.extract_group_sizes(data))
        result.update(self.extract_tests(data))
        result.update(self.extract_analyzer_data(data))
        if self.front:
            result = self.rename_passed(result)
        return result

    def report(self, data: ExperimentData) -> dict[str, Any]:
        result = {
            "splitter_id": self.get_splitter_id(data),
        }
        result.update(self.extract_data_from_analysis_tables(data))
        return result


class AADatasetReporter(OneAADictReporter):
    def report(self, data: ExperimentData):
        front_buffer = self.front
        self.front = False
        dict_report = super().report(data)
        self.front = front_buffer
        return self.convert_flat_dataset(dict_report)


class AAPassedReporter(Reporter):
    @staticmethod
    def _reformat_aa_score_table(table: Dataset) -> Dataset:
        if table.is_empty() or "pass" not in table.columns:
            return SmallDataset.create_empty()
            
        pass_dict = table.get_values(0, "pass")
        if not isinstance(pass_dict, dict) or not pass_dict:
            return SmallDataset.create_empty()

        index_map = {}
        test_names = set()
        for k, v in pass_dict.items():
            parts = k.split(ID_SPLIT_SYMBOL)
            row_idx = f"{parts[0]}{ID_SPLIT_SYMBOL}{parts[-1]}" if len(parts) >= 3 else k
            test_name = parts[1] if len(parts) >= 3 else "StatsTTest"
            test_names.add(test_name)
            index_map.setdefault(row_idx, {})[test_name] = 1 if v else 0

        indices = list(index_map.keys())
        rows = [{t: index_map[idx].get(t, 0) for t in test_names} for idx in indices]
        if not rows: return SmallDataset.create_empty()

        ds = SmallDataset.from_dict(rows, roles={t: InfoRole() for t in test_names})
        ds.index = indices
        return ds

    @staticmethod
    def _reformat_best_split_table(table: Dataset) -> Dataset:
        pass_cols = [c for c in table.columns if c.endswith("pass")]
        if not pass_cols or table.is_empty():
            return SmallDataset.create_empty()

        indices = []
        rows = []
        for i in range(len(table)):
            feat = str(table.get_values(i, "feature")) if "feature" in table.columns else f"R{i}"
            grp = str(table.get_values(i, "group")) if "group" in table.columns else ""
            indices.append(f"{feat}{ID_SPLIT_SYMBOL}{grp}")
            
            row = {}
            for col in pass_cols:
                val = table.get_values(i, col)
                name = col[:col.rfind("pass")-1].strip() if "pass" in col else col
                row[name] = 1 if str(val).strip().upper() in ["OK", "TRUE", "1", "YES", "True"] else 0
            rows.append(row)
            
        if not rows: return SmallDataset.create_empty()
        ds = SmallDataset.from_dict(rows, roles={c: InfoRole() for c in rows[0].keys()})
        ds.index = indices
        return ds

    def _detect_pass(self, analyzer_tables: dict[str, Dataset]):
        score_table = self._reformat_aa_score_table(analyzer_tables["aa score"])
        best_split_table = self._reformat_best_split_table(analyzer_tables["best split statistics"])

        if best_split_table.is_empty():
            return SmallDataset.create_empty(roles={"feature": InfoRole(), "group": InfoRole(), "result": StatisticRole()})

        t_data = best_split_table.to_dict()
        col_data = t_data.get("data", {}).get("data", {})
        idx_list = t_data.get("data", {}).get("index", list(range(len(next(iter(col_data.values()))))))

        records = []
        n_rows = len(idx_list)
        for i in range(n_rows):
            rec = {}
            if "feature" in col_data:
                rec["feature"] = col_data["feature"][i]
            else:
                rec["feature"] = str(idx_list[i]).split(ID_SPLIT_SYMBOL)[0]
            if "group" in col_data:
                rec["group"] = col_data["group"][i]
            else:
                rec["group"] = str(idx_list[i]).split(ID_SPLIT_SYMBOL)[-1]

            passed_any = False
            for col, vals in col_data.items():
                if col in ("feature", "group"):
                    continue
                val = vals[i]
                rec[col] = val
                if val in (1, True, "OK", "True", 1.0):
                    passed_any = True
            rec["result"] = "OK" if passed_any else "NOT OK"
            records.append(rec)

        roles = {"feature": InfoRole(), "group": InfoRole(), "result": StatisticRole()}
        for col in col_data:
            if col not in ("feature", "group"):
                roles[col] = best_split_table.roles.get(col, StatisticRole())

        result = SmallDataset.from_dict(records, roles=roles)

        for col in result.columns:
            if col not in ("feature", "group", "result"):
                result.data[col] = result.data[col].astype(str).replace({
                    "1.0": "OK", "0.0": "NOT OK", "1": "OK", "0": "NOT OK",
                    "True": "OK", "False": "NOT OK"
                })
        return result

    def report(self, data: ExperimentData) -> Dataset:
        analyser_ids = data.get_ids("AAScoreAnalyzer", ExperimentDataEnum.analysis_tables)
        analyser_tables = {
            id_[id_.rfind(ID_SPLIT_SYMBOL) + 1 :]: data.analysis_tables[id_]
            for id_ in analyser_ids["AAScoreAnalyzer"][ExperimentDataEnum.analysis_tables.value]
        }
        if not analyser_tables.get("aa score") or analyser_tables["aa score"].is_empty():
            print("AA test cannot be performed as none of the analyzers passed")
            return None

        result = self._detect_pass(analyser_tables)
        if result is None or result.is_empty():
            return result

        stats_cols = ["feature", "group", "control mean", "test mean", "difference", "difference %"]
        diff_source = analyser_tables.get("best split statistics")
        if diff_source is not None and not diff_source.is_empty():
            available_stats = [c for c in stats_cols if c in diff_source.columns]
            differences = diff_source.select(available_stats) if available_stats else Dataset.create_empty()
        else:
            differences = Dataset.create_empty()

        if not differences.is_empty() and "feature" in result.columns and "group" in result.columns:
            try:
                result = result.merge(differences, on=["feature", "group"], how="left")
            except Exception:
                pass

        if not result.is_empty() and len(result.columns) > 0:
            final_cols = ["feature", "group"] + [c for c in result.columns if c not in ["feature", "group"]]
            try:
                result = result.select([c for c in final_cols if c in result.columns])
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


class AABestSplitReporter(Reporter):
    def report(self, data: ExperimentData):
        best_split_id = next(
            (c for c in data.additional_fields.columns if c.endswith("best")), None
        )
        if best_split_id is None:
            return data.ds

        markers = data.additional_fields.select([best_split_id])
        markers = markers.rename({best_split_id: "split"})
        return data.ds.merge(markers, left_index=True, right_index=True)
