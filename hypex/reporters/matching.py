from __future__ import annotations

from typing import Any, ClassVar

from ..analyzers.matching import MatchingAnalyzer
from ..comparators import KSTest, TTest, Chi2Test
from ..dataset import Dataset, ExperimentData
from ..ml import FaissNearestNeighbors
from ..reporters.abstract import DatasetReporter, DictReporter, TestDictReporter
from ..utils import (
    ID_SPLIT_SYMBOL,
    MATCHING_INDEXES_SPLITTER_SYMBOL,
    ExperimentDataEnum,
)


class MatchingDictReporter(DictReporter):
    def __init__(self, searching_class: type = MatchingAnalyzer):
        self.searching_class = searching_class
        super().__init__()

    @staticmethod
    def _convert_dataset_to_dict(data: Dataset) -> dict[str, Any]:
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
        indexes_id = data.get_ids(
            FaissNearestNeighbors, ExperimentDataEnum.additional_fields
        )[FaissNearestNeighbors.__name__][ExperimentDataEnum.additional_fields.value]
        return {
            f"indexes{ID_SPLIT_SYMBOL}{column.split(ID_SPLIT_SYMBOL)[3]}": MATCHING_INDEXES_SPLITTER_SYMBOL.join(
                str(i)
                for i in data.additional_fields[column].to_dict()["data"]["data"][
                    column
                ]
            )
            for column in indexes_id
        }

    def report(self, experiment_data: ExperimentData):
        result = {}
        result.update(self._extract_from_analyser(experiment_data))
        if self.searching_class == MatchingAnalyzer:
            result.update(self._extract_from_additional_fields(experiment_data))
        return result


class MatchingQualityDictReporter(TestDictReporter):
    tests: ClassVar[list] = [TTest, KSTest, Chi2Test]

    def report(self, data: ExperimentData) -> dict[str, Any]:
        return self.extract_tests(data)


class MatchingQualityDatasetReporter(MatchingQualityDictReporter):
    @classmethod
    def convert_flat_dataset(cls, data: dict) -> Dataset:
        struct_dict = cls._get_struct_dict(data)
        return cls._convert_struct_dict_to_dataset(struct_dict)

    def report(self, data: ExperimentData):
        front_buffer = self.front
        self.front = False
        dict_report = super().report(data)
        self.front = front_buffer
        result = self.convert_flat_dataset(dict_report)
        return result


class MatchingDatasetReporter(DatasetReporter):
    def __init__(self, searching_class: type = MatchingAnalyzer) -> None:
        self.dict_reporter = MatchingDictReporter(searching_class)
        super().__init__(self.dict_reporter)
