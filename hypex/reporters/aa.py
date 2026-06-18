from __future__ import annotations

import contextlib
from typing import Any, ClassVar

from ..comparators import Chi2Test, GroupDifference, GroupSizes, KSTest, TTest
from ..dataset import Dataset, ExperimentData, InfoRole, StatisticRole
from ..splitters import AASplitter, AASplitterWithStratification
from ..utils import ID_SPLIT_SYMBOL, ExperimentDataEnum, NotFoundInExperimentDataError
from .abstract import Reporter, TestDictReporter


class OneAADictReporter(TestDictReporter):
    tests: ClassVar[list] = [TTest, KSTest, Chi2Test]

    @staticmethod
    def convert_flat_dataset(data: dict) -> Dataset:
        struct_dict = OneAADictReporter._get_struct_dict(data)
        return OneAADictReporter._convert_struct_dict_to_dataset(struct_dict)

    @staticmethod
    def get_splitter_id(data: ExperimentData):
        for c in [AASplitter, AASplitterWithStratification]:
            with contextlib.suppress(NotFoundInExperimentDataError):
                return data.get_one_id(c, ExperimentDataEnum.additional_fields)

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
        result = {}
        for ind in table.index:
            splitted_index = ind.split(ID_SPLIT_SYMBOL)
            row_index = f"{splitted_index[0]}{ID_SPLIT_SYMBOL}{splitted_index[-1]}"
            value = table.get_values(ind, "pass")
            if row_index not in result:
                result[row_index] = {splitted_index[1]: value}
            else:
                result[row_index][splitted_index[1]] = value
        result = Dataset.from_dict(result, roles={}).transpose() * 1
        return result

    @staticmethod
    def _reformat_best_split_table(table: Dataset) -> Dataset:
        passed = table.loc[:, [c for c in table.columns if (c.endswith("pass"))]]
        new_index = table.apply(
            lambda x: f"{x['feature']}{ID_SPLIT_SYMBOL}{x['group']}",
            {"index": InfoRole()},
            axis=1,
        )
        passed.index = new_index.get_values(column="index")
        passed = passed.rename(
            names={c: c[: c.rfind("pass") - 1] for c in passed.columns}
        )
        passed = passed.replace("OK", 1).replace("NOT OK", 0)
        passed = passed.astype({c: int for c in passed.columns}, errors="ignore")
        return passed

    def _detect_pass(self, analyzer_tables: dict[str, Dataset]):
        score_table = self._reformat_aa_score_table(analyzer_tables["aa score"])
        best_split_table = self._reformat_best_split_table(
            analyzer_tables["best split statistics"]
        )
        resume_table = score_table * best_split_table
        resume_table = resume_table.apply(
            lambda x: "OK" if x.sum() > 0 else "NOT OK",
            axis=1,
            role={"result": StatisticRole()},
        )
        result = score_table.merge(
            best_split_table,
            suffixes=(" aa test", " best split"),
            left_index=True,
            right_index=True,
        )
        result = result.merge(resume_table, left_index=True, right_index=True)
        result.roles = {c: r.__class__(str) for c, r in result.roles.items()}
        result = (
            result.replace(0, "NOT OK")
            .replace(1, "OK")
            .replace("0", "NOT OK")
            .replace("1", "OK")
        )
        splitted_index = [str(i).split(ID_SPLIT_SYMBOL) for i in result.index]
        result.add_column([i[0] for i in splitted_index], role={"feature": InfoRole()})
        result.add_column([i[1] for i in splitted_index], role={"group": InfoRole()})
        result.index = range(len(splitted_index))
        return result

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
        if not analyser_tables["aa score"]:
            print("AA test cannot be performed as none of the analyzers passed")
            return None
        result = self._detect_pass(analyser_tables)
        stats_cols = [
            "feature",
            "group",
            "control mean",
            "test mean",
            "difference",
            "difference %",
        ]
        differences = analyser_tables["best split statistics"].loc[
            :,
            [
                col
                for col in stats_cols
                if col in analyser_tables["best split statistics"].columns
            ],
        ]
        result = result.merge(differences, on=["feature", "group"], how="left")
        result = result[
            ["feature", "group"]
            + [c for c in result.columns if c not in ["feature", "group"]]
        ]
        numeric_cols = ["control mean", "test mean", "difference", "difference %"]
        for col in numeric_cols:
            result.data[col] = result.data[col].astype(float).round(6)
        return result


class AABestSplitReporter(Reporter):
    def report(self, data: ExperimentData):
        best_split_id = next(
            (c for c in data.additional_fields.columns if c.endswith("best")), []
        )
        markers = data.additional_fields.loc[:, best_split_id]
        markers = markers.rename({markers.columns[0]: "split"})
        return data.ds.merge(markers, left_index=True, right_index=True)
