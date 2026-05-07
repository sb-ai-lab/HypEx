from __future__ import annotations

from typing import Any

from ..dataset import Dataset, ExperimentData
from .aa import OneAADictReporter
from .abstract import DatasetReporter


class HomoDictReporter(OneAADictReporter):
    def report(self, data: ExperimentData) -> dict[str, Any]:
        return self.extract_data_from_analysis_tables(data)


class HomoDatasetReporter(DatasetReporter):
    def __init__(self):
        super().__init__(dict_reporter=HomoDictReporter(front=False))

    @staticmethod
    def convert_to_dataset(data: dict) -> dict[str, Dataset] | Dataset:
        return HomoDictReporter.convert_flat_dataset(data)
