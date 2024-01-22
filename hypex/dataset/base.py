from abc import ABC
from typing import Any, Union, Sequence, Iterable
from pandas import DataFrame


class DatasetSeletor(ABC):

    @staticmethod
    def select_dataset(data):
        if isinstance(data, DataFrame):
            return PandasDataset(data)
        return 0

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __repr__(self):
        pass

    def __add__(self, other):
        pass


class PandasDataset(DatasetSeletor):

    def __init__(self, data: DataFrame):
        self.data = data

    def _get_column_index(self,
                          column_name: Union[Sequence[str], str]) -> Union[int, Sequence[int]]:
        idx = self.data.columns.get_loc(column_name) if isinstance(column_name, str) \
            else self.data.columns.get_indexer(column_name)
        return idx

    def __getitem__(self, item):
        if isinstance(item, slice) or isinstance(item, int):
            return self.data.iloc[item]
        if isinstance(item, str) or isinstance(item, list):
            return self.data[item]
        raise KeyError("No such column or row")

    def __len__(self):
        return len(self.data)

    def __setitem__(self, key: str, value: Iterable):
        self.data[key] = value
        return self.data

    def __repr__(self):
        return self.data.__repr__()

    def apply(self, *args, **kwargs):
        return self.data.apply(*args, **kwargs)


