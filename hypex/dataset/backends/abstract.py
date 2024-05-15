from abc import abstractmethod, ABC
from typing import Iterable, Any, Union, Callable, Sized, Optional, Dict, Sequence

from hypex.utils import FromDictType, FieldsType
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

    # Comparison methods:
    @abstractmethod
    def __eq__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __ne__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __le__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __lt__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __ge__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __gt__(self, other) -> Any:
        raise AbstractMethodError

    # Unary methods:
    @abstractmethod
    def __pos__(self) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __neg__(self) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __abs__(self) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __invert__(self) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __round__(self, ndigits: int = 0) -> Any:
        raise AbstractMethodError

    # Binary methods:
    @abstractmethod
    def __add__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __sub__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __mul__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __floordiv__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __div__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __truediv__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __mod__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __pow__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __and__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __or__(self, other) -> Any:
        raise AbstractMethodError

    # Right arithmetic methods:
    @abstractmethod
    def __radd__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __rsub__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __rmul__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __rfloordiv__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __rdiv__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __rtruediv__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __rmod__(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def __rpow__(self, other) -> Any:
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
    def mode(self, numeric_only: bool = False, dropna: bool = True) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def var(
        self, skipna: bool = True, ddof: int = 1, numeric_only: bool = False
    ) -> Any:
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
    def log(self) -> Any:
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
    def is_empty(self) -> Any:
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
    def corr(selfself, method="pearson", numeric_only=False) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def na_counts(self) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def dropna(self, how: str = "any", subset: Union[str, Iterable[str]] = []) -> Any:
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
    def merge(
        self,
        right: Any,
        on: FieldsType = "",
        left_on: FieldsType = "",
        right_on: FieldsType = "",
        left_index: bool = False,
        right_index: bool = False,
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def drop(self, labels: FieldsType = "", axis: int = 1) -> Any:
        raise AbstractMethodError
