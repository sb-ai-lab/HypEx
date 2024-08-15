from typing import Dict, Any

from hypex.comparators import GroupDifference, GroupSizes, TTest, KSTest, Chi2Test
from hypex.dataset import (
    ExperimentData,
    Dataset,
    InfoRole,
    TreatmentRole,
    StatisticRole,
)
from hypex.splitters import AASplitter, AASplitterWithStratification
from hypex.utils import (
    ExperimentDataEnum,
    ID_SPLIT_SYMBOL,
    NotFoundInExperimentDataError,
)
from .abstract import DictReporter, Reporter


class OneAADictReporter(DictReporter):
    @staticmethod
    def rename_passed(data: Dict[str, bool]):
        return {
            c: ("NOT OK" if v else "OK") if "pass" in c else v for c, v in data.items()
        }

    @staticmethod
    def _get_struct_dict(data: Dict):
        # TODO: rewrite to recursion?
        dict_result = {}
        for key, value in data.items():
            if ID_SPLIT_SYMBOL in key:
                key_split = key.split(ID_SPLIT_SYMBOL)
                if key_split[2] in ("pass", "p-value"):
                    if key_split[0] not in dict_result:
                        dict_result[key_split[0]] = {
                            key_split[3]: {key_split[1]: {key_split[2]: value}}
                        }
                    elif key_split[3] not in dict_result[key_split[0]]:
                        dict_result[key_split[0]][key_split[3]] = {
                            key_split[1]: {key_split[2]: value}
                        }
                    elif key_split[1] not in dict_result[key_split[0]][key_split[3]]:
                        dict_result[key_split[0]][key_split[3]][key_split[1]] = {
                            key_split[2]: value
                        }
                    else:
                        dict_result[key_split[0]][key_split[3]][key_split[1]][
                            key_split[2]
                        ] = value
        return dict_result

    @staticmethod
    def _convert_struct_dict_to_dataset(data: Dict) -> Dataset:
        result = []
        for feature, groups in data.items():
            for group, tests in groups.items():
                t_values = {"feature": feature, "group": group}
                for test, values in tests.items():
                    t_values[f"{test} pass"] = values["pass"]
                    t_values[f"{test} p-value"] = values["p-value"]
                result.append(t_values)
        result = [OneAADictReporter.rename_passed(d) for d in result]
        return Dataset.from_dict(
            result,
            roles={"feature": InfoRole(), "group": TreatmentRole()},
        )

    @staticmethod
    def convert_flat_dataset(data: Dict) -> Dataset:
        struct_dict = OneAADictReporter._get_struct_dict(data)
        return OneAADictReporter._convert_struct_dict_to_dataset(struct_dict)

    @staticmethod
    def get_splitter_id(data: ExperimentData):
        for c in [AASplitter, AASplitterWithStratification]:
            try:
                return data.get_one_id(c, ExperimentDataEnum.additional_fields)
            except NotFoundInExperimentDataError:
                pass  # The splitting was done by another class

    def extract_group_difference(self, data: ExperimentData) -> Dict[str, Any]:
        group_difference_ids = data.get_ids(GroupDifference)[GroupDifference.__name__][
            ExperimentDataEnum.analysis_tables.value
        ]
        return self._extract_from_comparators(data, group_difference_ids)

    def extract_group_sizes(self, data: ExperimentData) -> Dict[str, Any]:
        group_sizes_id = data.get_one_id(GroupSizes, ExperimentDataEnum.analysis_tables)
        return self._extract_from_comparators(data, [group_sizes_id])

    def extract_tests(self, data: ExperimentData) -> Dict[str, Any]:
        test_ids = data.get_ids(
            [TTest, KSTest, Chi2Test], searched_space=ExperimentDataEnum.analysis_tables
        )
        result = {}
        for class_, ids in test_ids.items():
            result.update(
                self._extract_from_comparators(
                    data, ids[ExperimentDataEnum.analysis_tables.value]
                )
            )
        return {k: v for k, v in result.items() if "pass" in k or "p-value" in k}

    def extract_analyzer_data(self, data: ExperimentData) -> Dict[str, Any]:
        analyzer_id = data.get_one_id(
            "OneAAStatAnalyzer", ExperimentDataEnum.analysis_tables
        )
        return self.extract_from_one_row_dataset(data.analysis_tables[analyzer_id])

    def extract_data_from_analysis_tables(self, data: ExperimentData) -> Dict[str, Any]:
        result = {}
        result.update(self.extract_group_difference(data))
        result.update(self.extract_group_sizes(data))
        result.update(self.extract_tests(data))
        result.update(self.extract_analyzer_data(data))
        if self.front:
            result = self.rename_passed(result)
        return result

    def report(self, data: ExperimentData) -> Dict[str, Any]:
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
        passed = table.loc[:, [c for c in table.columns if c.endswith("pass")]]
        new_index = table.apply(
            lambda x: f"{x['feature']}{ID_SPLIT_SYMBOL}{x['group']}",
            {"index": InfoRole()},
            axis=1,
        )
        passed.index = new_index.get_values(column="index")
        passed = passed.rename(
            names={c: c[: c.rfind("pass") - 1] for c in passed.columns}
        )
        passed.roles = {c: r.__class__(int) for c, r in passed.roles.items()}
        passed = passed.replace("OK", 1).replace("NOT OK", 0)
        return passed

    def _detect_pass(self, analyzer_tables: Dict[str, Dataset]):
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
        result = result.replace(0, "NOT OK").replace(1, "OK")
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
        return self._detect_pass(analyser_tables)


class AABestSplitReporter(Reporter):
    def report(self, data: ExperimentData):
        best_split_id = next(
            (c for c in data.additional_fields.columns if c.endswith("best")), []
        )
        markers = data.additional_fields.loc[:, best_split_id]
        markers = markers.rename({markers.columns[0]: "split"})
        return data.ds.merge(markers, left_index=True, right_index=True)
