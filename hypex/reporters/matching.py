from typing import Dict, Any, Union

from ..dataset import (
    ExperimentData,
    Dataset,
    InfoRole,
    TreatmentRole,
)

from ..ml import FaissNearestNeighbors
from ..reporters.abstract import DictReporter, DatasetReporter
from ..utils import (
    ExperimentDataEnum,
    ID_SPLIT_SYMBOL,
    MATCHING_INDEXES_SPLITTER_SYMBOL,
)

from ..analyzers.matching import MatchingAnalyzer
from .abstract import DatasetReporter, DictReporter
from ..comparators import TTest, KSTest


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


class MatchingQualityDictReporter(DictReporter):

    @staticmethod
    def _get_struct_dict(data: Dict):
        dict_result = {}
        for key, value in data.items():
            if ID_SPLIT_SYMBOL in key:
                key_split = key.split(ID_SPLIT_SYMBOL)
                if key_split[2] in ("pass", "p-value"):
                    if key_split[0] not in dict_result:
                        dict_result[key_split[0]] = {
                            key_split[3]: {key_split[1]: {key_split[2]: value}}
                        }
                    elif key_split[3] not in dict_result[key_split[0]]:
                        dict_result[key_split[0]][key_split[3]] = {
                            key_split[1]: {key_split[2]: value}
                        }
                    elif key_split[1] not in dict_result[key_split[0]][key_split[3]]:
                        dict_result[key_split[0]][key_split[3]][key_split[1]] = {
                            key_split[2]: value
                        }
                    else:
                        dict_result[key_split[0]][key_split[3]][key_split[1]][
                            key_split[2]
                        ] = value
        return dict_result

    @staticmethod
    def _convert_struct_dict_to_dataset(data: Dict) -> Dataset:
        result = []
        for feature, groups in data.items():
            for group, tests in groups.items():
                t_values = {"feature": feature, "group": group}
                for test, values in tests.items():
                    t_values[f"{test} pass"] = values["pass"]
                    t_values[f"{test} p-value"] = values["p-value"]
                result.append(t_values)
        return Dataset.from_dict(
            result,
            roles={"feature": InfoRole(), "group": TreatmentRole()},
        )

    def extract_tests(self, data: ExperimentData) -> Dict[str, Any]:
        test_ids = data.get_ids(
            [TTest, KSTest], searched_space=ExperimentDataEnum.analysis_tables
        )
        result = {}
        for class_, ids in test_ids.items():
            result.update(
                self._extract_from_comparators(
                    data, ids[ExperimentDataEnum.analysis_tables.value]
                )
            )
        return {k: v for k, v in result.items() if "pass" in k or "p-value" in k}

    def extract_data_from_analysis_tables(self, data: ExperimentData) -> Dict[str, Any]:
        result = {}
        result.update(self.extract_tests(data))
        return result

    def report(self, data: ExperimentData) -> Dict[str, Any]:
        result = {}
        result.update(self.extract_data_from_analysis_tables(data))
        return result


class MatchingQualityDatasetReporter(MatchingQualityDictReporter):

    @staticmethod
    def convert_flat_dataset(data: Dict) -> Dataset:
        struct_dict = MatchingQualityDictReporter._get_struct_dict(data)
        return MatchingQualityDictReporter._convert_struct_dict_to_dataset(struct_dict)

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
