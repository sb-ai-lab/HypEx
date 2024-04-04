from typing import Dict, Any

from hypex.dataset.dataset import ExperimentData
from hypex.reporters.reporters import DictReporter
from hypex.splitters.aa import AASplitter
from hypex.utils.constants import ID_SPLIT_SYMBOL
from hypex.utils.enums import ExperimentDataEnum
from hypex.comparators.comparators import GroupDifference


class AADictReporter(DictReporter):
    def get_random_state(self, data: ExperimentData):
        aa_splitter_id = data._get_one_id(
            AASplitter, ExperimentDataEnum.additional_fields
        )
        aa_id = aa_splitter_id.split(ID_SPLIT_SYMBOL)[1]
        return int(aa_id) if aa_id.isdigit() else None

    def extract_group_difference(self, data: ExperimentData) -> Dict[str, Any]:
        aa_splitter_id = data.get_ids(GroupDifference)[GroupDifference][ExperimentDataEnum.analysis_tables.value]
        t_data = data.analysis_tables[aa_splitter_id[0]]
        for aid in aa_splitter_id[1:]:
            t_data = t_data.append(data.analysis_tables[aid])

        
        group_difference = t_data.to_dict()["data"]
        group_difference = [
            {
                f"{group} {group_difference['index'][i]}": group_difference["data"][
                    group
                ][i]
                for i in range(len(group_difference["index"]))
            }
            for group in group_difference["data"]
        ]
        result = group_difference[0]
        for i in range(1, len(group_difference)):
            result.update(group_difference[i])
        return result

    

    def extract_data_from_analysis_tables(self, data: ExperimentData) -> Dict[str, Any]:
        group_difference = self.extract_group_difference(data)
        return group_difference

    def report(self, data: ExperimentData) -> Dict[str, Any]:
        result = {
            "random_state": self.get_random_state(data),
        }
        result.update(self.extract_data_from_analysis_tables(data))
        return result
