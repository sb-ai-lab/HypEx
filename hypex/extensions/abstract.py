from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from ..dataset import ABCRole, Dataset
from ..dataset.backends import PandasDataset, SparkDataset
from ..dataset.dataset import DatasetAdapter
from ..utils.errors import AbstractMethodError


class Extension(ABC):
    @staticmethod
    def result_to_dataset(
        result: Any, roles: ABCRole | dict[str, ABCRole], small: bool=True
    ) -> Dataset:
        return DatasetAdapter.to_dataset(result, roles=roles,small=small)


class CompareExtension(Extension, ABC):
    def calc(self, data: Dataset, other: Dataset | None = None, **kwargs):
        return super().calc(data=data, other=other, **kwargs)


class MLExtension(Extension):
    @abstractmethod
    def fit(self, X, Y=None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X, **kwargs):
        raise NotImplementedError

    def calc(
        self,
        data: Dataset,
        mode: Literal["auto", "fit", "predict"] | None = None,
        **kwargs,
    ):
        if mode in ["auto", "fit"]:
            return self.fit(data, **kwargs)
        return self.predict(data, **kwargs)
        # return super().calc(data=data, **kwargs)
