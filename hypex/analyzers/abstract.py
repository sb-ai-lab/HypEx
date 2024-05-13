import warnings
from abc import ABC

from hypex.dataset import Dataset
from hypex.executor import Executor


class Analyzer(Executor, ABC):
    @staticmethod
    def calc(data: Dataset, **kwargs) -> Dataset:
        warnings.warn(f"Meaningless for {Analyzer.__name__}")
        return data
