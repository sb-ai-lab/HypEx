from abc import ABC, abstractmethod
from typing import Union, Sequence, Iterable

import pandas as pd
from pandas import DataFrame


def select_dataset(data):
    if isinstance(data, DataFrame):
        return PandasDataset(data)
    if isinstance(data, str):
        check_data = check_file_extension(data)
        if check_data is not None:
            return PandasDataset(check_data)
    return None


def check_file_extension(file_path):
    read_functions = {"csv": pd.read_csv, "xlsx": pd.read_excel, "json": pd.read_json}
    extension = file_path.split(".")[-1].lower()
    if extension in read_functions:
        read_function = read_functions[extension]
        return read_function(file_path)
    return None


class DatasetBase(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        pass

    @abstractmethod
    def apply(self, *args, **kwargs):
        pass

    @abstractmethod
    def map(self, *args, **kwargs):
        pass


class PandasDataset(DatasetBase):
    def __init__(self, data: DataFrame):
        self.data = data

    def _get_column_index(
        self, column_name: Union[Sequence[str], str]
    ) -> Union[int, Sequence[int]]:
        idx = (
            self.data.columns.get_loc(column_name)
            if isinstance(column_name, str)
            else self.data.columns.get_indexer(column_name)
        )[0]
        return idx

    def __getitem__(self, item):
        if isinstance(item, slice) or isinstance(item, int):
            return self.data.iloc[item]
        if isinstance(item, str) or isinstance(item, list):
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
