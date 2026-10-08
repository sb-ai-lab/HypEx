from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from ..dataset import ABCRole, Dataset
from ..dataset.dataset import DatasetAdapter
from ..utils import BackendsEnum


class Extension(ABC):
    BACKEND_MAPPING: ClassVar[dict[BackendsEnum, str]] = {
        BackendsEnum.pandas: "_calc_pandas",
        BackendsEnum.spark: "_calc_spark",
    }

    @staticmethod
    def result_to_dataset(
        result: Any, roles: ABCRole | dict[str, ABCRole], small: bool = True
    ) -> Dataset:
        return DatasetAdapter.to_dataset(result, roles=roles, small=small)

    def calc(self, data: Dataset, *args, **kwargs):
        backend = data.backend_type
        if backend not in self.BACKEND_MAPPING:
            raise ValueError(
                f"{type(self).__name__} has no implementation for backend {backend!r}. "
                f"Registered: {list(self.BACKEND_MAPPING)}"
            )
        return getattr(self, self.BACKEND_MAPPING[backend])(data, *args, **kwargs)

    def _calc_pandas(self, data: Dataset, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} has no pandas implementation")

    def _calc_spark(self, data: Dataset, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} has no spark implementation")


class CompareExtension(Extension, ABC):
    def calc(self, data: Dataset, other: Dataset | None = None, **kwargs):
        raise NotImplementedError


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
