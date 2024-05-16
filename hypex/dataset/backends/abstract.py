from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Sized, Union

from hypex.utils import FromDictTypes
from hypex.utils.errors import AbstractMethodError


class DatasetBackendNavigation(ABC):
    @property
    def name(self) -> str:
        return str(self.__class__.__name__).lower().replace("backend", "")

    @property
    @abstractmethod
    def index(self):
        raise AbstractMethodError

    @property
    @abstractmethod
    def columns(self):
        raise AbstractMethodError

    @abstractmethod
    def from_dict(
        self, data: FromDictTypes, index: Optional[Union[Iterable, Sized]] = None
    ):
        raise AbstractMethodError

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        raise AbstractMethodError

    @abstractmethod
    def __getitem__(self, item) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __len__(self) -> int:
        raise AbstractMethodError

    @abstractmethod
    def __repr__(self):
        raise AbstractMethodError

    @abstractmethod
    def create_empty(
        self,
        index: Optional[Iterable] = None,
        columns: Optional[Iterable[str]] = None,
    ):
        raise AbstractMethodError

    @abstractmethod
    def _get_column_index(
        self, column_name: Union[Sequence[str], str]
    ) -> Union[int, Sequence[int]]:
        raise AbstractMethodError

    @abstractmethod
    def _get_column_type(self, column_name: str) -> str:
        raise AbstractMethodError

    @abstractmethod
    def _update_column_type(self, column_name: str, type_name: str):
        raise AbstractMethodError

    @abstractmethod
    def add_column(
        self,
        data: Union[Sequence],
        name: str,
        index: Optional[Sequence] = None,
    ):
        raise AbstractMethodError

    @abstractmethod
    def append(self, other, index: bool = False) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def loc(self, values: Iterable) -> Iterable:
        raise AbstractMethodError

    @abstractmethod
    def iloc(self, values: Iterable) -> Iterable:
        raise AbstractMethodError


class DatasetBackendCalc(DatasetBackendNavigation, ABC):
    @abstractmethod
    def mean(self) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def max(self) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def min(self) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def count(self) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def sum(self) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def agg(self, func) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def apply(self, func: Callable, **kwargs) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def map(self, func: Callable, **kwargs) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def unique(self) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def isin(self, values: Iterable) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def groupby(self, by: Union[str, Iterable[str]], **kwargs) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def sort_index(self, **kwargs) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def sort_values(
        self, by: Union[str, List[str]], ascending: bool = True, **kwargs
    ) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def fillna(self, values, method, **kwargs) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def dot(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def transpose(self, names) -> Any:
        raise AbstractMethodError

    def shuffle(self, random_state) -> Any:
        raise AbstractMethodError
