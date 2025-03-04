from typing import Dict, Any, Union

from ..dataset import ExperimentData, Dataset
from .aa import OneAADictReporter
from .abstract import DatasetReporter


class HomoDictReporter(OneAADictReporter):
    def report(self, data: ExperimentData) -> Dict[str, Any]:
        return self.extract_data_from_analysis_tables(data)


class HomoDatasetReporter(DatasetReporter):
    def __init__(self):
        super().__init__(dict_reporter=HomoDictReporter(front=False))

    @staticmethod
    def convert_to_dataset(data: Dict) -> Union[Dict[str, Dataset], Dataset]:
        return HomoDictReporter.convert_flat_dataset(data)
