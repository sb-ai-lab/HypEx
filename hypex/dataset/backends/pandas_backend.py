from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence, Sized

import numpy as np
import pandas as pd  # type: ignore

from ...utils import FromDictTypes, MergeOnError, ScalarType
from .abstract import DatasetBackendCalc, DatasetBackendNavigation


class PandasNavigation(DatasetBackendNavigation):
    @staticmethod
    def _read_file(filename: str | Path) -> pd.DataFrame:
        file_extension = Path(filename).suffix
        if file_extension == ".csv":
            return pd.read_csv(filename)
        elif file_extension == ".xlsx":
            return pd.read_excel(filename)
        else:
            raise ValueError(f"Unsupported file extension {file_extension}")

    def __init__(self, data: pd.DataFrame | dict | str | pd.Series | None = None):
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = pd.DataFrame(data)
        elif isinstance(data, dict):
            if "index" in data.keys():
                self.data = pd.DataFrame(data=data["data"], index=data["index"])
            else:
                self.data = pd.DataFrame(data=data["data"])
        elif isinstance(data, str):
            self.data = self._read_file(data)
        else:
            self.data = pd.DataFrame()

    def __getitem__(self, item):
        if isinstance(item, (slice, int)):
            return self.data.iloc[item]
        if isinstance(item, (str, list)):
            return self.data[item]
        if isinstance(item, pd.DataFrame):
            if len(item.columns) == 1:
                return self.data[item.iloc[:, 0]]
            else:
                return self.data[item]
        raise KeyError("No such column or row")

    def __len__(self):
        return len(self.data)

    @staticmethod
    def __magic_determine_other(other) -> Any:
        if isinstance(other, PandasDataset):
            return other.data
        else:
            return other

    # comparison operators:
    def __eq__(self, other) -> Any:
        return self.data == self.__magic_determine_other(other)

    def __ne__(self, other) -> Any:
        return self.data != self.__magic_determine_other(other)

    def __le__(self, other) -> Any:
        return self.data <= self.__magic_determine_other(other)

    def __lt__(self, other) -> Any:
        return self.data < self.__magic_determine_other(other)

    def __ge__(self, other) -> Any:
        return self.data >= self.__magic_determine_other(other)

    def __gt__(self, other) -> Any:
        return self.data > self.__magic_determine_other(other)

    # Unary operations:
    def __pos__(self) -> Any:
        return +self.data

    def __neg__(self) -> Any:
        return -self.data

    def __abs__(self) -> Any:
        return abs(self.data)

    def __invert__(self) -> Any:
        return ~self.data

    def __round__(self, ndigits: int = 0) -> Any:
        return round(self.data, ndigits)

    # Binary operations:
    def __add__(self, other) -> Any:
        return self.data + self.__magic_determine_other(other)

    def __sub__(self, other) -> Any:
        return self.data - self.__magic_determine_other(other)

    def __mul__(self, other) -> Any:
        return self.data * self.__magic_determine_other(other)

    def __floordiv__(self, other) -> Any:
        return self.data // self.__magic_determine_other(other)

    def __div__(self, other) -> Any:
        return self.data / self.__magic_determine_other(other)

    def __truediv__(self, other) -> Any:
        return self.data / self.__magic_determine_other(other)

    def __mod__(self, other) -> Any:
        return self.data % self.__magic_determine_other(other)

    def __pow__(self, other) -> Any:
        return self.data ** self.__magic_determine_other(other)

    def __and__(self, other) -> Any:
        return self.data & self.__magic_determine_other(other)

    def __or__(self, other) -> Any:
        return self.data | self.__magic_determine_other(other)

    # Right arithmetic operators:
    def __radd__(self, other) -> Any:
        return self.__magic_determine_other(other) + self.data

    def __rsub__(self, other) -> Any:
        return self.__magic_determine_other(other) - self.data

    def __rmul__(self, other) -> Any:
        return self.__magic_determine_other(other) * self.data

    def __rfloordiv__(self, other) -> Any:
        return self.__magic_determine_other(other) // self.data

    def __rdiv__(self, other) -> Any:
        return self.__magic_determine_other(other) / self.data

    def __rtruediv__(self, other) -> Any:
        return self.__magic_determine_other(other) / self.data

    def __rmod__(self, other) -> Any:
        return self.__magic_determine_other(other) % self.data

    def __rpow__(self, other) -> Any:
        return self.__magic_determine_other(other) ** self.data

    def __repr__(self):
        return self.data.__repr__()

    def _repr_html_(self):
        return self.data._repr_html_()

    def create_empty(
        self,
        index: Iterable | None = None,
        columns: Iterable[str] | None = None,
    ):
        self.data = pd.DataFrame(index=index, columns=columns)
        return self

    @property
    def index(self):
        return self.data.index

    @property
    def columns(self):
        return self.data.columns

    @property
    def shape(self):
        return self.data.shape

    def _get_column_index(
        self, column_name: Sequence[str] | str
    ) -> int | Sequence[int]:
        return (
            self.data.columns.get_loc(column_name)
            if isinstance(column_name, str)
            else self.data.columns.get_indexer(column_name)
        )[0]

    def get_column_type(self, column_name: str) -> type | None:
        dtype = self.data.dtypes[column_name]
        if pd.api.types.is_integer_dtype(dtype):
            return int
        elif pd.api.types.is_float_dtype(dtype):
            return float
        elif (
            pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_object_dtype(dtype)
            or dtype.name == "category"
        ):
            return str
        elif pd.api.types.is_bool_dtype(dtype):
            return bool
        else:
            return None

    def astype(
        self, dtype: dict[str, type], errors: Literal["raise", "ignore"] = "raise"
    ) -> pd.DataFrame:
        return self.data.astype(dtype=dtype, errors=errors)

    def update_column_type(self, column_name: str, type_name: type):
        if self.data[column_name].isna().sum() == 0:
            self.data = self.data.astype({column_name: type_name})
        return self

    def add_column(
        self,
        data: Sequence,
        name: str | list[str],
        index: Sequence | None = None,
    ):
        if isinstance(name, list) and len(name) == 1:
            name = name[0]
        if isinstance(data, pd.DataFrame):
            data = data.values
        if len(self.data) != len(data):
            if isinstance(data[0], Iterable) and len(data[0]) == 1:
                data = data.squeeze()
            data = pd.Series(data)
        if index:
            self.data = self.data.join(
                pd.DataFrame(data, columns=[name], index=list(index))
            )
        else:
            self.data.loc[:, name] = data

    def append(self, other, reset_index: bool = False, axis: int = 0) -> pd.DataFrame:
        new_data = pd.concat([self.data] + [d.data for d in other], axis=axis)
        if reset_index:
            new_data = new_data.reset_index(drop=True)
        return new_data

    def from_dict(self, data: FromDictTypes, index: Iterable | Sized | None = None):
        if isinstance(data, dict):
            self.data = pd.DataFrame().from_records(data, columns=list(data.keys()))
        else:
            self.data = pd.DataFrame().from_records(data)
        if index is not None:
            self.data.index = index
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": {
                column: self.data[column].to_list() for column in self.data.columns
            },
            "index": list(self.index),
        }

    def to_records(self) -> list[dict]:
        return self.data.to_dict(orient="records")

    def loc(self, items: Iterable) -> Iterable:
        data = self.data.loc[items]
        if not isinstance(data, Iterable) or isinstance(data, str):
            data = [data]
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    def iloc(self, items: Iterable) -> Iterable:
        data = self.data.iloc[items]
        if not isinstance(data, Iterable) or isinstance(data, str):
            data = [data]
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)


class PandasDataset(PandasNavigation, DatasetBackendCalc):
    @staticmethod
    def _convert_agg_result(result):
        if isinstance(result, pd.Series):
            result = result.to_frame()
        if result.shape[0] == 1 and result.shape[1] == 1:
            return float(result.loc[result.index[0], result.columns[0]])
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)

    def __init__(self, data: pd.DataFrame | dict | str | pd.Series | None = None):
        super().__init__(data)

    def get_values(
        self,
        row: str | None = None,
        column: str | None = None,
    ) -> Any:
        if (column is not None) and (row is not None):
            return self.data.loc[row, column]
        elif column is not None:
            result = self.data.loc[:, column]
        elif row is not None:
            result = self.data.loc[row, :]
        else:
            result = self.data
        return result.values.tolist()

    def iget_values(
        self,
        row: int | None = None,
        column: int | None = None,
    ) -> Any:
        if (column is not None) and (row is not None):
            return self.data.iloc[row, column]
        elif column is not None:
            result = self.data.iloc[:, column]
        elif row is not None:
            result = self.data.iloc[row, :]
        else:
            result = self.data
        return result.values.tolist()

    def apply(self, func: Callable, **kwargs) -> pd.DataFrame:
        single_column_name = kwargs.pop("column_name")
        result = self.data.apply(func, **kwargs)
        if not isinstance(result, pd.DataFrame):
            result = result.to_frame(name=single_column_name)
        return result

    def map(self, func: Callable, **kwargs) -> pd.DataFrame:
        return self.data.map(func, **kwargs)

    def is_empty(self) -> bool:
        return self.data.empty

    def unique(self):
        return {column: self.data[column].unique() for column in self.data.columns}

    def nunique(self, dropna: bool = True):
        return {column: self.data[column].nunique() for column in self.data.columns}

    def groupby(self, by: str | Iterable[str], **kwargs) -> list[tuple]:
        groups = self.data.groupby(by=by, observed=False, **kwargs)
        return list(groups)

    def agg(self, func: str | list, **kwargs) -> pd.DataFrame | float:
        func = func if isinstance(func, (list, dict)) else [func]
        result = self.data.agg(func, **kwargs)
        return self._convert_agg_result(result)

    def max(self) -> pd.DataFrame | float:
        return self.agg(["max"])

    def idxmax(self) -> pd.DataFrame | float:
        return self.agg(["idxmax"])

    def min(self) -> pd.DataFrame | float:
        return self.agg(["min"])

    def count(self) -> pd.DataFrame | float:
        return self.agg(["count"])

    def sum(self) -> pd.DataFrame | float:
        return self.agg(["sum"])

    def mean(self) -> pd.DataFrame | float:
        return self.agg(["mean"])

    def mode(
        self, numeric_only: bool = False, dropna: bool = True
    ) -> pd.DataFrame | float:
        return self.data.mode(numeric_only=numeric_only, dropna=dropna)

    def var(
        self, skipna: bool = True, ddof: int = 1, numeric_only: bool = False
    ) -> pd.DataFrame | float:
        return self.agg(["var"], skipna=skipna, ddof=ddof, numeric_only=numeric_only)

    def log(self) -> pd.DataFrame:
        np_data = np.log(self.data.to_numpy())
        return pd.DataFrame(np_data, columns=self.data.columns)

    def std(self, skipna: bool = True, ddof: int = 1) -> pd.DataFrame | float:
        return self.agg(["std"], skipna=skipna, ddof=ddof)

    def cov(self):
        return self.data.cov(ddof=1)

    def quantile(self, q: float = 0.5) -> pd.DataFrame:
        if isinstance(q, list) and len(q) > 1:
            return self.data.quantile(q=q)
        return self.agg(func="quantile", q=q)

    def coefficient_of_variation(self) -> pd.DataFrame | float:
        data = (self.data.std() / self.data.mean()).to_frame().T
        data.index = ["cv"]
        if data.shape[0] == 1 and data.shape[1] == 1:
            return float(data.loc[data.index[0], data.columns[0]])
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    def sort_index(self, ascending: bool = True, **kwargs) -> pd.DataFrame:
        return self.data.sort_index(ascending=ascending, **kwargs)

    def corr(
        self,
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
        numeric_only: bool = False,
    ) -> pd.DataFrame | float:
        return self.data.corr(method=method, numeric_only=numeric_only)

    def isna(self) -> pd.DataFrame:
        return self.data.isna()

    def sort_values(
        self, by: str | list[str], ascending: bool = True, **kwargs
    ) -> pd.DataFrame:
        return self.data.sort_values(by=by, ascending=ascending, **kwargs)

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> pd.DataFrame:
        return self.data.value_counts(
            normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
        ).reset_index()

    def fillna(
        self,
        values: ScalarType | dict[str, ScalarType] | None = None,
        method: Literal["bfill", "ffill"] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        if method is not None:
            if method == "bfill":
                return self.data.bfill(**kwargs)
            elif method == "ffill":
                return self.data.ffill(**kwargs)
            else:
                raise ValueError(f"Wrong fill method: {method}")

        return self.data.fillna(value=values, **kwargs)

    def na_counts(self) -> pd.DataFrame | int:
        data = self.data.isna().sum().to_frame().T
        data.index = ["na_counts"]
        if data.shape[0] == 1 and data.shape[1] == 1:
            return int(data.loc[data.index[0], data.columns[0]])
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    def dot(self, other: PandasDataset) -> pd.DataFrame:
        result = self.data.dot(other.data)
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)

    def dropna(
        self,
        how: Literal["any", "all"] = "any",
        subset: str | Iterable[str] | None = None,
        axis: Literal["index", "rows", "columns"] | int = 0,
    ) -> pd.DataFrame:
        return self.data.dropna(how=how, subset=subset, axis=axis)

    def transpose(self, names: Sequence[str] | None = None) -> pd.DataFrame:
        result = self.data.transpose()
        if names is not None:
            result.columns = names
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)

    def sample(
        self,
        frac: float | None = None,
        n: int | None = None,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        return self.data.sample(n=n, frac=frac, random_state=random_state)

    def select_dtypes(
        self,
        include: str | None = None,
        exclude: str | None = None,
    ) -> pd.DataFrame:
        return self.data.select_dtypes(include=include, exclude=exclude)

    def isin(self, values: Iterable) -> Iterable[bool]:
        return self.data.isin(values)

    def merge(
        self,
        right: PandasDataset,
        on: str | None = None,
        left_on: str | None = None,
        right_on: str | None = None,
        left_index: bool | None = None,
        right_index: bool | None = None,
        suffixes: tuple[str, str] = ("_x", "_y"),
        how: Literal["left", "right", "inner", "outer", "cross"] = "inner",
    ) -> pd.DataFrame:
        for on_ in [on, left_on, right_on]:
            if on_ and (
                on_ not in [*self.columns, *right.columns]
                if isinstance(on_, str)
                else any(c not in [*self.columns, *right.columns] for c in on_)
            ):
                raise MergeOnError(on_)

        if not all(
            [
                on,
                left_on,
                right_on,
            ]
        ) and all([left_index is None, right_index is None]):
            left_index = True
            right_index = True

        return self.data.merge(
            right=right.data,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            suffixes=suffixes,
            how=how,
        )

    def drop(self, labels: str = "", axis: int = 1) -> pd.DataFrame:
        return self.data.drop(labels=labels, axis=axis)

    def filter(
        self,
        items: list | None = None,
        like: str | None = None,
        regex: str | None = None,
        axis: int = 0,
    ) -> pd.DataFrame:
        return self.data.filter(items=items, like=like, regex=regex, axis=axis)

    def rename(self, columns: dict[str, str]) -> pd.DataFrame:
        return self.data.rename(columns=columns)

    def replace(
        self, to_replace: Any = None, value: Any = None, regex: bool = False
    ) -> pd.DataFrame:
        if isinstance(to_replace, pd.DataFrame) and len(to_replace.columns) == 1:
            to_replace = to_replace.iloc[:, 0]
        elif isinstance(to_replace, pd.Series):
            to_replace = to_replace.to_list()
        elif isinstance(to_replace, dict):
            return self.data.replace(to_replace=to_replace, regex=regex)
        return self.data.replace(to_replace=to_replace, value=value, regex=regex)

    def reindex(self, labels: str = "", fill_value: str | None = None) -> pd.DataFrame:
        return self.data.reindex(labels, fill_value=fill_value)
