from abc import ABC, abstractmethod
from typing import Union, Any, Dict

from hypex.dataset import Dataset
from hypex.dataset.backends import PandasDataset
from hypex.utils.errors import AbstractMethodError 
from hypex.utils import FieldKeyTypes
from hypex.utils.adapter import Adapter 
from hypex.dataset import ABCRole




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
    def result_to_dataset(result: Any, roles: Union[ABCRole, Dict[FieldKeyTypes, ABCRole]]) -> Dataset:
        return Adapter.to_dataset(result, roles=roles)


class CompareExtension(Extension, ABC):
    def calc(self, data: Dataset, other: Union[Dataset, None] = None, **kwargs):
        return super().calc(data=data, other=other, **kwargs)
