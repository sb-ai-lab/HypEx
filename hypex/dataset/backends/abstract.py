from abc import abstractmethod, ABC
from typing import Iterable, Any, Union, Callable, Sized, Optional, Dict, Sequence

from hypex.utils import FromDictType
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
        self, data: FromDictType, index: Optional[Union[Iterable, Sized]] = None
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
    def _create_empty(
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
    def nunique(self, dropna: bool = True) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def isin(self, values: Iterable) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def groupby(self, by: Union[str, Iterable[str]], **kwargs) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def std(self) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def coefficient_of_variation(self) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def value_counts(self, normalize: bool = False, sort: bool = True, ascending: bool = False, dropna: bool = True) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def na_counts(self) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def dropna(self, how: ["any", "all"] = "any",  subset: Union[str, Iterable[str]] = None) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def isna(self) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def quantile(self, q: float = 0.5) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def select_dtypes(self, include: Any = None, exclude: Any = None) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def merge(self, right, on=None, left_on=None, right_on=None, left_index=False, right_index=False, suffixes=('_x', '_y')) -> Any:
        raise AbstractMethodError


    @abstractmethod
    def drop(self, labels: Any = None, axis: int = 1) -> Any:
        raise AbstractMethodError
