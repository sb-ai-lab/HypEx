from abc import ABC, abstractmethod
from typing import Dict, Any

from hypex.dataset.dataset import ExperimentData, Dataset

class Reporter(ABC):
    @abstractmethod
    def report(self, data: ExperimentData):
        raise NotImplementedError

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
        raise NotImplementedError