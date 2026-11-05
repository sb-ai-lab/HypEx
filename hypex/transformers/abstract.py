from __future__ import annotations

from abc import abstractmethod

from ..dataset import ABCRole, Dataset, ExperimentData, FeatureRole
from ..executor import Calculator, MLExecutor
from ..utils import AbstractMethodError


class Transformer(Calculator):
    @property
    def _is_transformer(self):
        return True

    @staticmethod
    @abstractmethod
    def _inner_function(data: Dataset, *args, **kwargs) -> Dataset:
        raise AbstractMethodError

    @classmethod
    def calc(cls, data: Dataset, **kwargs):
        return cls._inner_function(data, **kwargs)

    def execute(self, data: ExperimentData) -> ExperimentData:
        data = data.copy(data=self.calc(data=data.ds))
        return data

class MLtransformer(MLExecutor):
    def __init__(
        self,
        target_roles: ABCRole | list[ABCRole] | None = None,
        key = "",
        **kwargs
    ):
        super().__init__(key, **kwargs)
        self.target_roles = target_roles or FeatureRole()

    @property
    def _is_transformer(self):
        return True
