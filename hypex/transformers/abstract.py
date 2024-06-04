from abc import abstractmethod
from typing import Any

from hypex.dataset import Dataset
from hypex.executor import Calculator
from hypex.utils import AbstractMethodError


class Transformer(Calculator):
    @property
    def __is_transformer(self):
        return True

    @classmethod
    @abstractmethod
    def calc(cls, data: Dataset, **kwargs):
        return cls._inner_function(data, **kwargs)

    @staticmethod
    @abstractmethod
    def _inner_function(data: Dataset, **kwargs) -> Any:
        raise AbstractMethodError
