from typing import Dict, Any

from hypex.analyzers import OneAAStatAnalyzer
from hypex.comparators import GroupDifference, GroupSizes, TTest, KSTest, Chi2Test
from hypex.dataset import (
    ExperimentData,
    Dataset,
    InfoRole,
    TreatmentRole,
    TargetRole,
    StatisticRole,
)
from hypex.splitters import AASplitter, AASplitterWithStratification
from hypex.utils import (
    ExperimentDataEnum,
    ID_SPLIT_SYMBOL,
    NotFoundInExperimentDataError,
)
from .abstract import DictReporter


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
        group_difference_ids = data.get_ids(GroupDifference)[GroupDifference][
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
            OneAAStatAnalyzer, ExperimentDataEnum.analysis_tables
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
