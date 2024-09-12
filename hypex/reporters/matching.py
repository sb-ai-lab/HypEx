from typing import Dict, Any

from hypex.analyzers.matching import MatchingAnalyzer
from hypex.dataset.dataset import ExperimentData, Dataset
from hypex.ml import FaissNearestNeighbors
from hypex.reporters.abstract import DictReporter, DatasetReporter
from hypex.utils import (
    ExperimentDataEnum,
    ID_SPLIT_SYMBOL,
    MATCHING_INDEXES_SPLITTER_SYMBOL,
)


class MatchingDictReporter(DictReporter):
    def __init__(self, searching_class: type = MatchingAnalyzer):
        self.searching_class = searching_class
        super().__init__()

    @staticmethod
    def _convert_dataset_to_dict(data: Dataset) -> Dict[str, Any]:
        dict_data = data.to_dict()["data"]
        indexes = dict_data["index"]
        df = dict_data["data"]
        result = {}
        for key, values in df.items():
            for index, value in zip(indexes, values):
                result[f"{key}{ID_SPLIT_SYMBOL}{index}"] = value
        return result

    def _extract_from_analyser(self, data: ExperimentData):
        analyzer_id = data.get_one_id(
            self.searching_class, ExperimentDataEnum.analysis_tables
        )
        return self._convert_dataset_to_dict(data.analysis_tables[analyzer_id])

    @staticmethod
    def _extract_from_additional_fields(data: ExperimentData):
        indexes_id = data.get_one_id(
            FaissNearestNeighbors, ExperimentDataEnum.additional_fields
        )
        return {
            "indexes": MATCHING_INDEXES_SPLITTER_SYMBOL.join(
                str(i)
                for i in data.additional_fields[indexes_id].to_dict()["data"]["data"][
                    indexes_id
                ]
            )
        }

    def report(self, experiment_data: ExperimentData):
        result = {}
        result.update(self._extract_from_analyser(experiment_data))
        if self.searching_class == MatchingAnalyzer:
            result.update(self._extract_from_additional_fields(experiment_data))
        return result


class MatchingDatasetReporter(DatasetReporter):

    def __init__(self, searching_class: type = MatchingAnalyzer) -> None:
        self.dict_reporter = MatchingDictReporter(searching_class)
        super().__init__(self.dict_reporter)
