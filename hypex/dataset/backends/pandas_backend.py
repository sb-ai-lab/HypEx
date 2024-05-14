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

from hypex.dataset.backends.abstract import DatasetBackendCalc, DatasetBackendNavigation
from hypex.utils import FromDictTypes


class PandasNavigation(DatasetBackendNavigation):
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
            self.data = pd.DataFrame()

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

    @property
    def index(self):
        return self.data.index

    @property
    def columns(self):
        return self.data.columns

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
            self.data = self.data.join(
                pd.DataFrame(data, columns=[name], index=list(index))
            )
        else:
            self.data.loc[:, name] = data

    def _create_empty(
        self,
        index: Optional[Iterable] = None,
        columns: Optional[Iterable[str]] = None,
    ):
        self.data = pd.DataFrame(index=index, columns=columns)
        return self

    def from_dict(
        self, data: FromDictTypes, index: Union[Iterable, Sized, None] = None
    ):
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

    def append(self, other, index: bool = False) -> pd.DataFrame:
        new_data = pd.concat([self.data, other.data])
        if index:
            new_data.reset_index()
        return new_data

    def loc(self, items: Iterable) -> Iterable:
        data = self.data.loc[items]
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    def iloc(self, items: Iterable) -> Iterable:
        data = self.data.iloc[items]
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)


class PandasDataset(PandasNavigation, DatasetBackendCalc):

    def __init__(self, data: Union[pd.DataFrame, Dict, str, pd.Series] = None):
        super().__init__(data)

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

    def mean(self) -> Union[pd.DataFrame, float]:
        return self.agg(["mean"])

    def max(self) -> Union[pd.DataFrame, float]:
        return self.agg(["max"])

    def min(self) -> Union[pd.DataFrame, float]:
        return self.agg(["min"])

    def count(self) -> Union[pd.DataFrame, float]:
        return self.agg(["count"])

    def sum(self) -> Union[pd.DataFrame, float]:
        return self.agg(["sum"])

    def agg(self, func: Union[str, List]) -> Union[pd.DataFrame, float]:
        func = func if isinstance(func, List) else [func]
        result = self.data.agg(func)
        if result.shape[0] == 1 and result.shape[1] == 1:
            return float(result.loc[result.index[0], result.columns[0]])
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)

    def sort_index(self, ascending: bool = True, **kwargs) -> pd.DataFrame:
        return self.data.sort_index(**kwargs)

    def sort_values(
        self, by: Union[str, List[str]], ascending: bool = True, **kwargs
    ) -> pd.DataFrame:
        return self.data.sort_values(by=by, ascending=ascending, **kwargs)

    def fillna(self, values, method, **kwargs) -> pd.DataFrame:
        return self.data.fillna(values, method=method, **kwargs)
