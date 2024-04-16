import warnings
from abc import ABC

from hypex.dataset.dataset import Dataset
from hypex.experiments.base import ComplexExecutor


class Analyzer(ComplexExecutor, ABC):
    def calc(self, data: Dataset) -> Dataset:
        warnings.warn(f"Meaningless for {self.__class__.__name__}")
        return data
