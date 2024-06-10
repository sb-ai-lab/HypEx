from typing import Dict, Any

from hypex.analyzers import OneAAStatAnalyzer
from hypex.comparators import GroupDifference, GroupSizes
from hypex.dataset import ExperimentData
from hypex.splitters import AASplitter
from hypex.utils import ExperimentDataEnum
from hypex.utils import ID_SPLIT_SYMBOL
from .abstract import DictReporter


class AADictReporter(DictReporter):
    @staticmethod
    def get_random_state(data: ExperimentData):
        aa_splitter_id = data.get_one_id(
            AASplitter, ExperimentDataEnum.additional_fields
        )
        aa_id = aa_splitter_id.split(ID_SPLIT_SYMBOL)[1]
        return int(aa_id) if aa_id.isdigit() else None

    def extract_group_difference(self, data: ExperimentData) -> Dict[str, Any]:
        group_difference_ids = data.get_ids(GroupDifference)[GroupDifference][
            ExperimentDataEnum.analysis_tables.value
        ]
        return self._extract_from_comparators(data, group_difference_ids)

    def extract_group_sizes(self, data: ExperimentData) -> Dict[str, Any]:
        group_sizes_id = data.get_one_id(GroupSizes, ExperimentDataEnum.analysis_tables)
        return self._extract_from_comparators(data, [group_sizes_id])

    def extract_analyzer_data(self, data: ExperimentData) -> Dict[str, Any]:
        analyzer_id = data.get_one_id(
            OneAAStatAnalyzer, ExperimentDataEnum.analysis_tables
        )
        result = self.extract_from_one_row_dataset(data.analysis_tables[analyzer_id])
        if self.front:
            result = {
                c: ("NOT OK" if v else "OK") if c.endswith("pass") else v
                for c, v in result.items()
            }
        return result

    def extract_data_from_analysis_tables(self, data: ExperimentData) -> Dict[str, Any]:
        result = {}
        result.update(self.extract_group_difference(data))
        result.update(self.extract_group_sizes(data))
        result.update(self.extract_analyzer_data(data))
        return result

    def report(self, data: ExperimentData) -> Dict[str, Any]:
        result = {
            "random_state": self.get_random_state(data),
        }
        result.update(self.extract_data_from_analysis_tables(data))
        return result
