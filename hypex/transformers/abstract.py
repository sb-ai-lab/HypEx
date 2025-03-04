from abc import abstractmethod

from ..dataset import Dataset, ExperimentData
from ..executor import Calculator
from ..utils import AbstractMethodError


class Transformer(Calculator):
    @property
    def _is_transformer(self):
        return True

    @staticmethod
    @abstractmethod
    def _inner_function(data: Dataset, **kwargs) -> Dataset:
        raise AbstractMethodError

    @classmethod
    def calc(cls, data: Dataset, **kwargs):
        return cls._inner_function(data, **kwargs)

    def execute(self, data: ExperimentData) -> ExperimentData:
        data = data.copy(data=self.calc(data=data.ds))
        return data
