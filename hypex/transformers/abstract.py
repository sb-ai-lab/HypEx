from abc import abstractmethod
from typing import Any

from hypex.dataset import Dataset, ExperimentData
from hypex.executor import Calculator
from hypex.utils import AbstractMethodError


class Transformer(Calculator):
    @property
    def _is_transformer(self):
        return True

    @staticmethod
    @abstractmethod
    def _inner_function(data: Dataset, **kwargs) -> Dataset:
        raise AbstractMethodError

    @classmethod
    @abstractmethod
    def calc(cls, data: Dataset, **kwargs):
        return cls._inner_function(data, **kwargs)

    def execute(self, data: ExperimentData) -> ExperimentData:
        data = data.copy(data=self.calc(data.ds))
        return data

    @staticmethod
    def _list_unification(roles):
        return [roles] if isinstance(roles, ABCRole) else roles