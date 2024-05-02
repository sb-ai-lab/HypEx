from pathlib import Path
from typing import (
    Sequence,
    Union,
    Iterable,
    List,
    Dict,
    Tuple,
    Sized,
    Callable,
    Optional,
    Any,
)

import pandas as pd  # type: ignore

from hypex.dataset.backends.abstract import DatasetBackendCalc
from hypex.utils import FromDictType


class PandasDataset(DatasetBackendCalc):
    @staticmethod
    def _read_file(filename: Union[str, Path]) -> pd.DataFrame:
        file_extension = Path(filename).suffix
        if file_extension == ".csv":
            return pd.read_csv(filename)
        elif file_extension == ".xlsx":
            return pd.read_excel(filename)
        else:
            raise ValueError(f"Unsupported file extension {file_extension}")

    def __init__(self, data: Union[pd.DataFrame, Dict, str, pd.Series] = None):
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = pd.DataFrame(data)
        elif isinstance(data, Dict):
            if "index" in data.keys():
                self.data = pd.DataFrame(data=data["data"], index=data["index"])
            else:
                self.data = pd.DataFrame(data=data["data"])
        elif isinstance(data, str):
            self.data = self._read_file(data)
        else:
            self.data = None

    def __getitem__(self, item):
        if isinstance(item, (slice, int)):
            return self.data.iloc[item]
        if isinstance(item, (str, list)):
            return self.data[item]
        raise KeyError("No such column or row")

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.data.__repr__()

    def _create_empty(
        self,
        index: Optional[Iterable] = None,
        columns: Optional[Iterable[str]] = None,
    ):
        self.data = pd.DataFrame(index=index, columns=columns)
        return self

    def _get_column_index(
        self, column_name: Union[Sequence[str], str]
    ) -> Union[int, Sequence[int]]:
        return (
            self.data.columns.get_loc(column_name)
            if isinstance(column_name, str)
            else self.data.columns.get_indexer(column_name)
        )[0]

    def _get_column_type(self, column_name: str) -> str:
        return str(self.data.dtypes[column_name])

    def _update_column_type(self, column_name: str, type_name: str):
        self.data[column_name] = self.data[column_name].astype(type_name)
        return self

    def add_column(
        self,
        data: Union[Sequence],
        name: str,
        index: Optional[Sequence] = None,
    ):
        if index:
            self.data[name] = [None] * len(data)
            for i, value in zip(index, data):
                self.data[name][i] = value
        else:
            self.data[name] = data

    def append(self, other, index: bool = False) -> pd.DataFrame:
        new_data = pd.concat([self.data, other.data])
        if index:
            new_data.reset_index()
        return new_data

    @property
    def index(self):
        return self.data.index

    @property
    def columns(self):
        return self.data.columns

    def from_dict(self, data: FromDictType, index: Union[Iterable, Sized, None] = None):
        self.data = pd.DataFrame().from_records(data)
        if index:
            self.data.index = index
        return self

    def to_dict(self) -> Dict[str, Any]:
        data = self.data.to_dict()
        for key, value in data.items():
            data[key] = list(data[key].values())
        index = list(self.index)
        return {"data": data, "index": index}

    def apply(self, func: Callable, **kwargs) -> pd.DataFrame:
        return self.data.apply(func, **kwargs)

    def map(self, func: Callable, **kwargs) -> pd.DataFrame:
        return self.data.map(func, **kwargs)

    def unique(self):
        return [(column, self.data[column].unique()) for column in self.data.columns]

    def isin(self, values: Iterable) -> Iterable[bool]:
        return self.data.isin(values)

    def groupby(self, by: Union[str, Iterable[str]], **kwargs) -> List[Tuple]:
        groups = self.data.groupby(by, **kwargs)
        return list(groups)

    def loc(self, items: Iterable) -> Iterable:
        data = self.data.loc[items]
        return pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data

    def iloc(self, items: Iterable) -> Iterable:
        data = self.data.iloc[items]
        return pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data

    def mean(self) -> pd.DataFrame:
        return self.data.agg(["mean"])

    def max(self) -> pd.DataFrame:
        return self.data.agg(["max"])

    def min(self) -> pd.DataFrame:
        return self.data.agg(["min"])

    def count(self) -> pd.DataFrame:
        return self.data.agg(["count"])

    def sum(self) -> pd.DataFrame:
        return self.data.agg(["sum"])

    def agg(self, func) -> pd.DataFrame:
        return self.data.agg(func)
