from typing import Dict, Optional, Union

import pandas as pd
from pandas import DataFrame

from hypex.dataset.backends.pandas_backend import PandasDataset
from hypex.dataset.base import DatasetBase
from hypex.dataset.roles import ABCRole
from hypex.dataset.utils import parse_roles


def select_dataset(data):
    if isinstance(data, pd.DataFrame):
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


class Dataset(DatasetBase):
    def set_data(self, data: Union[DataFrame, str] = None, roles=None):
        self.roles = parse_roles(roles)
        self.data = select_dataset(data)

    def __init__(
        self,
        data: Union[DataFrame, str, None] = None,
        roles: Optional[Dict[ABCRole, Union[list[str], str]]] = None,
    ):
        self.data = None
        self.roles = None
        self.set_data(data, roles)

    def __repr__(self):
        return self.data.__repr__()

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

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
        return self.data.apply(func, axis, raw, result_type, args, by_row, **kwargs)

    def map(self, func, na_action=None, **kwargs):
        return self.data.map(func, na_action, **kwargs)


class ExperimentData(Dataset):
    def __init__(self, data):
        self.additional_fields = DataFrame()
        self.stats_fields = DataFrame()
        self.analysis_tables = {}  # I think, we will know about analysis and stats,
        # that user want to make, but I don't understand their format
        self._create_fields(data)

    def _create_fields(self, data: pd.DataFrame):
        self.stats_fields.index = list(data.columns)
        self.additional_fields.index = data.index
        # can add keys for analysis_tables and columns for stats_fields

    def add_to_analysis_tables(self, key: str, data: pd.DataFrame):
        self.analysis_tables[key] = data

    def add_to_stats_fields(self, data: pd.DataFrame):
        self.stats_fields = self.stats_fields.join(data, on=self.stats_fields.index)

    def add_to_additional_fields(self, data: pd.DataFrame):
        self.additional_fields = self.additional_fields.join(
            data, on=self.additional_fields.index
        )

    def __str__(self):
        return "stats fields\n{}\nadditional fields\n{}\nanalysis tables\n{}".format(
            self.stats_fields.__str__(),
            self.additional_fields.__str__(),
            self.analysis_tables.__str__(),
        )

# как получать данные из stats_fields в формате [feature, stat]?
# пока идея только через loc. либо я могу хранить транспонированную матрицу, колонки - фичи, индексы - статистики
