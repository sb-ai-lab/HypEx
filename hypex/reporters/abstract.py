from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..dataset import Dataset, ExperimentData
from ..dataset.roles import InfoRole, ReportRole, TreatmentRole
from ..utils import ID_SPLIT_SYMBOL, ExperimentDataEnum
from ..utils.errors import AbstractMethodError


class Reporter(ABC):
    @abstractmethod
    def report(self, data: ExperimentData):
        raise AbstractMethodError


class DictReporter(Reporter, ABC):
    def __init__(self, front=True):
        self.front = front

    @staticmethod
    def extract_from_one_row_dataset(data: Dataset) -> dict[str, Any]:
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
        self, data: ExperimentData, comparator_ids: list[str]
    ) -> dict[str, Any]:
        result = {}
        for comparator_id in comparator_ids:
            result.update(self._extract_from_comparator(data, comparator_id))
        return result

    @abstractmethod
    def report(self, data: ExperimentData) -> dict:
        raise AbstractMethodError


class OnDictReporter(Reporter, ABC):
    def __init__(self, dict_reporter: DictReporter) -> None:
        self.dict_reporter = dict_reporter


class DatasetReporter(OnDictReporter):
    def report(self, data: ExperimentData) -> dict[str, Dataset] | Dataset:
        dict_result = self.dict_reporter.report(data)
        return self.convert_to_dataset(
            dict_result
        )  #   TODO: change to DatasetAdapter.to_dataset()

    @staticmethod
    def convert_to_dataset(data: dict) -> dict[str, Dataset] | Dataset:
        return Dataset.from_dict(roles={k: ReportRole() for k in data}, data=[data])


class TestDictReporter(DictReporter):
    @staticmethod
    def _get_struct_dict(data: dict):
        dict_result = {}
        for key, value in data.items():
            if ID_SPLIT_SYMBOL in key:
                key_split = key.split(ID_SPLIT_SYMBOL)
                if key_split[2] in ("pass", "p-value", "difference", "difference %", "control mean", "test mean"):
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
    def _convert_struct_dict_to_dataset(data: dict) -> Dataset:
        def rename_passed(data: dict[str, bool]):
            return {
                c: (
                    ("NOT OK" if (v is True or v == "True") else "OK")
                    if "pass" in c
                    else v
                )
                for c, v in data.items()
            }

        result = []
        for feature, groups in data.items():
            for group, tests in groups.items():
                t_values = {"feature": feature, "group": group}
                for test, values in tests.items():
                    if test == "GroupDifference":
                        t_values["control mean"] = values.get("control mean")
                        t_values["test mean"] = values.get("test mean")
                        t_values["difference"] = values.get("difference")
                        t_values["difference %"] = values.get("difference %")
                    else:
                        t_values[f"{test} pass"] = values.get("pass")
                        t_values[f"{test} p-value"] = values.get("p-value")
                result.append(t_values)
        result = [rename_passed(d) for d in result]
        return Dataset.from_dict(
            result,
            roles={"feature": InfoRole(), "group": TreatmentRole()},
        )

    def extract_tests(self, data: ExperimentData) -> dict[str, Any]:
        test_ids = data.get_ids(
            self.tests, searched_space=ExperimentDataEnum.analysis_tables
        )
        result = {}
        for class_, ids in test_ids.items():
            result.update(
                self._extract_from_comparators(
                    data, ids[ExperimentDataEnum.analysis_tables.value]
                )
            )
        return {k: v for k, v in result.items() if "pass" in k or "p-value" in k}
