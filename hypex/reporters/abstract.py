from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List

from hypex.dataset import ExperimentData, Dataset
from hypex.dataset.roles import ReportRole
from hypex.utils import ID_SPLIT_SYMBOL
from hypex.utils.errors import AbstractMethodError


class Reporter(ABC):
    @abstractmethod
    def report(self, data: ExperimentData):
        raise AbstractMethodError


class DictReporter(Reporter, ABC):
    def __init__(self, front=True):
        self.front = front

    @staticmethod
    def extract_from_one_row_dataset(data: Dataset) -> Dict[str, Any]:
        return {k: v[0] for k, v in data.to_dict()["data"]["data"].items()}

    def _extract_from_comparator(self, data: ExperimentData, comparator_id: str):
        result = {}
        field = comparator_id[comparator_id.rfind(ID_SPLIT_SYMBOL) + 1 :]
        executor_name = comparator_id[: comparator_id.find(ID_SPLIT_SYMBOL)]
        sep = " " if self.front else ID_SPLIT_SYMBOL
        analysis_dict = data.analysis_tables[comparator_id].to_dict()["data"]
        for i, index_value in enumerate(analysis_dict["index"]):
            for k, v in analysis_dict["data"].items():
                key = sep.join(
                    [field, executor_name, k, str(index_value)]
                    if field
                    else [executor_name, k, str(index_value)]
                )
                result[key] = v[i]
        return result

    def _extract_from_comparators(
        self, data: ExperimentData, comparator_ids: List[str]
    ) -> Dict[str, Any]:
        result = {}
        for comparator_id in comparator_ids:
            result.update(self._extract_from_comparator(data, comparator_id))
        return result

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
