from abc import ABC, abstractmethod
from typing import Dict, Any

from hypex.dataset import ExperimentData, Dataset
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
