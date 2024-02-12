from typing import Sequence, Union, Iterable

import pandas as pd

from hypex.dataset.base import DatasetBase


class PandasDataset(DatasetBase):
    def __init__(self, data: pd.DataFrame):
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

    def __setitem__(self, key: Union[str, slice], value: Iterable):
        self.data[key] = value

    def __repr__(self):
        return self.data.__repr__()

    def apply(self, *args, **kwargs):
        return self.data.apply(*args, **kwargs)

    def map(self, *args, **kwargs):
        return self.data.map(*args, **kwargs)
