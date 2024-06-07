from abc import ABC, abstractmethod
from typing import Dict, Any, Union

from hypex.dataset import ExperimentData, Dataset
from hypex.dataset.roles import ReportRole
from hypex.utils.errors import AbstractMethodError


class Reporter(ABC):
    @abstractmethod
    def report(self, data: ExperimentData):
        raise AbstractMethodError

    @staticmethod
    def extract_from_one_row_dataset(data: Dataset) -> Dict[str, Any]:
        return {k: v[0] for k, v in data.to_dict()["data"]["data"].items()}

    @staticmethod
    def _extract_from_comparators(data: Dataset) -> Dict[str, Any]:
        group_difference = data.to_dict()["data"]
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


class DictReporter(Reporter, ABC):
    @abstractmethod
    def report(self, data: ExperimentData) -> Dict:
        raise AbstractMethodError


class OnDictReporter(Reporter, ABC):
    def __init__(self, dict_reporter: DictReporter) -> None:
        self.dict_reporter = dict_reporter


class DatasetReporter(OnDictReporter):
    def report(self, data: ExperimentData) -> Union[Dict[str, Dataset], Dataset]:
        dict_result = self.dict_reporter.report(data)
        return self.convert_to_dataset(dict_result)

    @staticmethod
    def convert_to_dataset(data: Dict) -> Union[Dict[str, Dataset], Dataset]:
        return Dataset.from_dict(roles={k: ReportRole() for k in data}, data=[data])
