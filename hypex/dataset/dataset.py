from typing import Dict, Optional, Union

import pandas as pd
from pandas import DataFrame

from hypex.dataset.base import DatasetBase, select_dataset
from hypex.dataset.roles import ABCRole
from hypex.dataset.utils import parse_roles


class Dataset(DatasetBase):
    def set_data(self, data: Union[DataFrame, str] = None, roles=None):
        self.roles = parse_roles(roles)
        self.data = select_dataset(data)

    def __init__(
        self,
        data: Union[DataFrame, str, None] = None,
        roles: Optional[Dict[ABCRole, Union[list[str], str]]] = None,
    ):
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
        self.create_fields(data)

    def create_fields(self, data: pd.DataFrame):
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

    def __repr__(self):
        return self.additional_fields.__repr__()

# как получать данные из stats_fields в формате [feature, stat]?
# пока идея только через loc. либо я могу хранить транспонированную матрицу, колонки - фичи, индексы - статистики
