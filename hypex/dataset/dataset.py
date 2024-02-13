from typing import Dict, Optional, Union, List, Iterable

import pandas as pd
from pandas import DataFrame

from hypex.dataset.backends.pandas_backend import PandasDataset
from hypex.dataset.base import DatasetBase
from hypex.dataset.roles import ABCRole
from hypex.dataset.utils import parse_roles


def select_backend(data):
    if isinstance(data, pd.DataFrame):
        return PandasDataset(data)
    if isinstance(data, str):
        check_data = check_file_extension(data)
        if check_data is not None:
            return PandasDataset(check_data)


def check_file_extension(file_path):
    read_functions = {"csv": pd.read_csv, "xlsx": pd.read_excel, "json": pd.read_json}
    extension = file_path.split(".")[-1].lower()
    if extension in read_functions:
        read_function = read_functions[extension]
        return read_function(file_path)


class Dataset(DatasetBase):
    def set_data(self, data: Union[DataFrame, str] = None, roles=None):
        self.roles = parse_roles(roles)
        self.backend = select_backend(data)
        self.data = self.backend.data

    def __init__(
        self,
        data: Union[DataFrame, str, None] = None,
        roles: Optional[Dict[ABCRole, Union[List[str], str]]] = None,
    ):
        self.set_data(data, roles)

    def __repr__(self):
        return self.backend.__repr__()

    def __len__(self):
        return self.backend.__len__()

    def __getitem__(self, item):
        return self.backend.__getitem__(item)

    def __setitem__(self, key, value):
        self.backend.__setitem__(key, value)

    def get_columns_by_roles(
        self, roles: Union[ABCRole, Iterable[ABCRole]]
    ) -> List[str]:
        roles = roles if isinstance(roles, Iterable) else [roles]
        return [
            column
            for column, role in self.roles.items()
            if any(isinstance(role, r) for r in roles)
        ]

    def apply(
        self,
        func,
        axis=0,
        raw=False,
        result_type=None,
        args=(),
        by_row="compat",
        **kwargs,
    ):
        return self.backend.apply(func, axis, raw, result_type, args, by_row, **kwargs)

    def map(self, func, na_action=None, **kwargs):
        return self.backend.map(func, na_action, **kwargs)

    def unique(self):
        return self.backend.unique()

    def isin(self, values: Iterable) -> Iterable[bool]:
        raise NotImplementedError

    def groupby(self):
        raise NotImplementedError

    @property
    def index(self):
        return self.backend.index


class ExperimentData(Dataset):
    def __init__(self, data: Dataset):
        self.additional_fields = Dataset(DataFrame(index=data.index))
        self.stats_fields = Dataset(DataFrame(index=list(data.columns)))
        self.analysis_tables = {}

    def add_to_additional_fields(self, data: pd.DataFrame):
        self.additional_fields = self.additional_fields.data.join(data, how="left")

    def add_to_stats_fields(self, data: pd.DataFrame):
        self.stats_fields = self.stats_fields.data.join(data, how="left")

    def add_to_analysis_tables(
        self,
        key: str,
        data: pd.DataFrame,
        roles: Optional[Dict[ABCRole, Union[List[str], str]]] = None,
    ):
        self.analysis_tables[key] = Dataset(data, roles)
