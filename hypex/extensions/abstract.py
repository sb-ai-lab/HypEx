from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from ..dataset import ABCRole, Dataset
from ..dataset.backends import PandasDataset
from ..dataset.dataset import DatasetAdapter
from ..utils.errors import AbstractMethodError


class Extension(ABC):
    def __init__(self):
        self.BACKEND_MAPPING = {
            PandasDataset: self._calc_pandas,
        }

    @abstractmethod
    def _calc_pandas(self, data: Dataset, **kwargs):
        raise AbstractMethodError

    def calc(self, data: Dataset, **kwargs):
        return self.BACKEND_MAPPING[type(data.backend)](data=data, **kwargs)

    @staticmethod
    def result_to_dataset(result: Any, roles: ABCRole | dict[str, ABCRole]) -> Dataset:
        return DatasetAdapter.to_dataset(result, roles=roles)


class CompareExtension(Extension, ABC):
    def calc(self, data: Dataset, other: Dataset | None = None, **kwargs):
        return super().calc(data=data, other=other, **kwargs)


class MLExtension(Extension):
    #   TODO: add model
    def _calc_pandas(
        self,
        data: Dataset,
        mode: Literal["auto", "fit", "predict"] | None = None,
        **kwargs,
    ):
        if mode in ["auto", "fit"]:
            return self.fit(data, **kwargs)
        return self.predict(data, **kwargs)

    @abstractmethod
    def fit(self, X, Y=None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X, **kwargs):
        raise NotImplementedError

    def calc(
        self,
        data: Dataset,
        **kwargs,
    ):
        return super().calc(data=data, **kwargs)
