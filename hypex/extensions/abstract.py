from abc import ABC, abstractmethod
from typing import Union, Any, Dict, Optional, Literal

from ..dataset import Dataset, ABCRole
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
    def result_to_dataset(
        result: Any, roles: Union[ABCRole, Dict[str, ABCRole]]
    ) -> Dataset:
        return DatasetAdapter.to_dataset(result, roles=roles)


class CompareExtension(Extension, ABC):
    def calc(self, data: Dataset, other: Optional[Dataset] = None, **kwargs):
        return super().calc(data=data, other=other, **kwargs)


class MLExtension(Extension):
    #   TODO: add model
    def _calc_pandas(
        self,
        data: Dataset,
        test_data: Optional[Dataset] = None,
        mode: Optional[Literal["auto", "fit", "predict"]] = None,
        **kwargs
    ):
        if mode in ["auto", "fit"]:
            return self.fit(data, test_data, **kwargs)
        return self.predict(data)

    @abstractmethod
    def fit(self, X, Y=None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X, **kwargs):
        raise NotImplementedError

    def calc(
        self,
        data: Dataset,
        target_data: Union[Dataset, None] = None,
        test_data: Optional[Dataset] = None,
        **kwargs
    ):
        return super().calc(
            data=data, target_data=target_data, test_data=test_data, **kwargs
        )
