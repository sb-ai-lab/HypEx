from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Literal, Sequence, Sized

from ...utils import AbstractMethodError, FromDictTypes


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
    def from_dict(self, data: FromDictTypes, index: Iterable | Sized | None = None):
        raise AbstractMethodError

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        raise AbstractMethodError

    def to_records(self) -> list[dict]:
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
    def _repr_html_(self):
        raise AbstractMethodError

    @abstractmethod
    def create_empty(
        self,
        index: Iterable | None = None,
        columns: Iterable[str] | None = None,
    ):
        raise AbstractMethodError

    @abstractmethod
    def _get_column_index(
        self, column_name: Sequence[str] | str
    ) -> int | Sequence[int]:
        raise AbstractMethodError

    @abstractmethod
    def get_column_type(self, column_name: str) -> type | None:
        raise AbstractMethodError

    @abstractmethod
    def astype(
        self, dtype: dict[str, type], errors: Literal["raise", "ignore"] = "raise"
    ) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def update_column_type(self, column_name: str, type_name: type):
        raise AbstractMethodError

    @abstractmethod
    def add_column(
        self,
        data: Sequence,
        name: str,
        index: Sequence | None = None,
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
    def idxmax(self) -> Any:
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
    def get_values(
        self,
        row: str | None = None,
        column: str | None = None,
    ) -> Any:
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
    def groupby(self, by: str | Iterable[str], **kwargs) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def sort_index(self, **kwargs) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def sort_values(self, by: str | list[str], ascending: bool = True, **kwargs) -> Any:
        raise AbstractMethodError

    def std(self) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def coefficient_of_variation(self) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def corr(self, method="pearson", numeric_only=False) -> Any:
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
    def dropna(
        self,
        how: Literal["any", "all"] = "any",
        subset: str | Iterable[str] | None = None,
        axis: Literal["index", "rows", "columns"] | int = 0,
    ) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def isna(self) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def quantile(self, q: float = 0.5) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def select_dtypes(
        self, include: Any | None = None, exclude: Any | None = None
    ) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def merge(
        self,
        right: Any,
        on: str | None = None,
        left_on: str | None = None,
        right_on: str | None = None,
        left_index: bool = False,
        right_index: bool = False,
        suffixes: tuple[str, str] = ("_x", "_y"),
        how: Literal["left", "right", "inner", "outer", "cross"] = "inner",
    ) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def drop(
        self,
        labels: str | Sequence[str] | None = None,
        axis: int = 1,
    ) -> Any:
        raise AbstractMethodError

    def filter(
        self,
        items: list | None = None,
        like: str | None = None,
        regex: str | None = None,
        axis: int = 0,
    ) -> Any:
        return AbstractMethodError

    def fillna(self, values, method, **kwargs) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def dot(self, other) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def transpose(self, names) -> Any:
        raise AbstractMethodError

    @abstractmethod
    def rename(self, columns: dict[str, str]) -> Any:
        return AbstractMethodError

    @abstractmethod
    def replace(
        self, to_replace: Any = None, value: Any = None, regex: bool = False
    ) -> Any:
        raise AbstractMethodError
