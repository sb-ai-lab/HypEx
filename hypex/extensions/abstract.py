from abc import ABC, abstractmethod
from typing import Union

from hypex.dataset import Dataset
from hypex.dataset.backends import PandasDataset
from hypex.utils.errors import AbstractMethodError


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


class CompareExtension(Extension, ABC):
    def calc(self, data: Dataset, other: Union[Dataset, None] = None, **kwargs):
        return super().calc(data=data, other=other, **kwargs)
