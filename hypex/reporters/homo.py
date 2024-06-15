from typing import Dict, Any, Union

from hypex.dataset import (
    ExperimentData,
    Dataset,
    InfoRole,
    TreatmentRole,
    TargetRole,
    StatisticRole,
)
from .aa import AADictReporter
from .abstract import DatasetReporter
from ..utils import ID_SPLIT_SYMBOL


class HomoDictReporter(AADictReporter):
    def report(self, data: ExperimentData) -> Dict[str, Any]:
        return self.extract_data_from_analysis_tables(data)


class HomoDatasetReporter(DatasetReporter):
    def __init__(self):
        super().__init__(dict_reporter=HomoDictReporter(front=False))

    @staticmethod
    def _get_struct_dict(data: Dict):
        dict_result = {}
        for key, value in data.items():
            if ID_SPLIT_SYMBOL in key:
                key_split = key.split(ID_SPLIT_SYMBOL)
                if key_split[1] in ("pass", "p-value"):
                    if key_split[0] not in dict_result:
                        dict_result[key_split[0]] = {
                            key_split[2]: {key_split[1]: value}
                        }
                    elif key_split[2] in dict_result[key_split[0]]:
                        dict_result[key_split[0]][key_split[2]][key_split[1]] = value
                    else:
                        dict_result[key_split[0]][key_split[2]] = {key_split[1]: value}
        return dict_result

    @staticmethod
    def _convert_struct_dict_to_dataset(data: Dict) -> Dataset:
        result = []
        for feature, groups in data.items():
            result.extend(
                {
                    "feature": feature,
                    "group": group,
                    "pass": values["pass"],
                    "p-value": values["p-value"],
                }
                for group, values in groups.items()
            )
        result = [HomoDictReporter.rename_passed(d) for d in result]
        return Dataset.from_dict(
            result,
            roles={
                "feature": InfoRole(),
                "group": TreatmentRole(),
                "pass": TargetRole(),
                "p-value": StatisticRole(),
            },
        )

    @staticmethod
    def convert_to_dataset(data: Dict) -> Union[Dict[str, Dataset], Dataset]:
        struct_dict = HomoDatasetReporter._get_struct_dict(data)
        return HomoDatasetReporter._convert_struct_dict_to_dataset(struct_dict)
