from pathlib import Path
from typing import Sequence, Union, Iterable, List, Dict

import pandas as pd

from hypex.dataset.base import DatasetBase


class PandasDataset(DatasetBase):

    def __init__(self, data: Union[pd.DataFrame, Dict, str] = None):
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, Dict):
            self.data = pd.DataFrame(
                data=data["data"], index=data["index"], columns=data["columns"]
            )
        elif isinstance(data, str):
            self.data = self._read_file(data)

    @staticmethod
    def _read_file(filename: str) -> pd.DataFrame:
        file_extension = Path(filename).suffix
        if file_extension == ".csv":
            return pd.read_csv(filename)
        elif file_extension == ".xlsx":
            return pd.read_excel(filename)
        else:
            raise ValueError(f"Unsupported file extension {file_extension}")

    def _get_column_index(
        self, column_name: Union[Sequence[str], str]
    ) -> Union[int, Sequence[int]]:
        return (
            self.data.columns.get_loc(column_name)
            if isinstance(column_name, str)
            else self.data.columns.get_indexer(column_name)
        )[0]

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

    def _create_empty(self, index=None, columns=None):
        self.data = pd.DataFrame(index=index, columns=columns)
        return self

    def from_dict(self, data: List[Dict]):
        self.data = pd.DataFrame().from_records(data)
        return self

    def to_dict(self):
        columns = list(self.columns)
        index = list(self.index)
        data = self.data.values.tolist()
        return {"columns": columns, "index": index, "data": data}

    def add_column(self, data, name):
        self.data[name] = data

    def apply(self, func, **kwargs):
        return self.data.apply(func, **kwargs)

    def map(self, func, **kwargs):
        return self.data.map(func, **kwargs)

    def unique(self):
        return self.data.unique()

    @property
    def index(self):
        return self.data.index

    @property
    def columns(self):
        return self.data.columns

    def isin(self, values: Iterable) -> Iterable[bool]:
        return self.data.isin(values)

    def groupby(self, by, axis, **kwargs):
        return self.data.groupby(by, axis, **kwargs)

    def loc(self, items: Iterable) -> Iterable:
        return self.data.loc[items]

    def iloc(self, items: Iterable) -> Iterable:
        return self.data.iloc[items]
