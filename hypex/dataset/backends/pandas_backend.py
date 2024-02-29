from typing import Sequence, Union, Iterable, List, Dict

import pandas as pd

from hypex.dataset.base import DatasetBase


class PandasDataset(DatasetBase):

    def __init__(self, data: pd.DataFrame = None):
        self.data = data

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
        return self.data

    def _create_empty(self, index=None, columns=None):
        self.data = pd.DataFrame(index=index, columns=columns)
        return self

    def from_dict(self, data: List[Dict]):
        self.data = pd.DataFrame().from_records(data)
        return self

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
