from typing import Dict, Any

from hypex.analyzers import OneAAStatAnalyzer
from hypex.comparators import GroupDifference, GroupSizes, TTest, KSTest, Chi2Test
from hypex.dataset import ExperimentData
from hypex.splitters import AASplitter
from hypex.utils import ExperimentDataEnum
from hypex.utils import ID_SPLIT_SYMBOL
from .abstract import DictReporter


class AADictReporter(DictReporter):
    @staticmethod
    def get_splitter_id(data: ExperimentData):
        return data.get_one_id(AASplitter, ExperimentDataEnum.additional_fields)

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

    @staticmethod
    def rename_passed(data: Dict[str, bool]):
        return {
            c: ("NOT OK" if v else "OK") if "pass" in c else v for c, v in data.items()
        }

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
