from typing import Dict, Optional, Union, List, Iterable, Any

import pandas as pd
from pandas import DataFrame

from hypex.dataset.backends.pandas_backend import PandasDataset
from hypex.dataset.base import DatasetBase
from hypex.dataset.roles import ABCRole
from hypex.dataset.utils import parse_roles


def check_file_extension(file_path):
    read_functions = {"csv": pd.read_csv, "xlsx": pd.read_excel, "json": pd.read_json}
    extension = file_path.split(".")[-1].lower()
    if extension in read_functions:
        read_function = read_functions[extension]
        return read_function(file_path)


class Dataset(DatasetBase):

    class Locker:
        def __init__(self, backend):
            self.backend = backend

        def __getitem__(self, item):
            return self.backend.loc(item)

    class ILocker:
        def __init__(self, backend):
            self.backend = backend

        def __getitem__(self, item):
            return self.backend.iloc(item)

    def set_data(self, data: Union[DataFrame, str] = None, roles=None):
        self.roles = parse_roles(roles)
        self._backend = self._select_backend(data)
        self.data = self._backend.data
        self.loc = self.Locker(self._backend)
        self.iloc = self.ILocker(self._backend)

    def __init__(
        self,
        data: Union[DataFrame, str, None] = None,
        roles: Optional[Dict[ABCRole, Union[List[str], str]]] = None,
    ):
        self.roles = None
        self._backend = None
        self.data = None
        self.loc = None
        self.iloc = None
        self.set_data(data, roles)

    def __repr__(self):
        return self.data.__repr__()

    def __len__(self):
        return self._backend.__len__()

    def __getitem__(self, item):
        return self._backend.__getitem__(item)

    @staticmethod
    def _select_backend(data):
        if isinstance(data, pd.DataFrame):
            return PandasDataset(data)
        if isinstance(data, str):
            check_data = check_file_extension(data)
            if check_data is not None:
                return PandasDataset(check_data)
        return None

    def get_columns_by_roles(
        self, roles: Union[ABCRole, Iterable[ABCRole]]
    ) -> List[str]:
        roles = roles if isinstance(roles, Iterable) else [roles]
        return [
            column
            for column, role in self.roles.items()
            if any(isinstance(role, r) for r in roles)
        ]

    def add_column(self, data, role: Dict):
        self.roles.update(role)
        self._backend.add_column(data, list(role.items())[0][0])

    def _create_empty(self, indexes=None, columns=None):
        indexes = [] if indexes is None else indexes
        columns = [] if columns is None else columns
        self._backend = self._backend._create_empty(indexes, columns)
        self.data = self._backend.data
        return self

    def apply(self, func, axis=0, **kwargs):
        return self._backend.apply(func=func, axis=axis, **kwargs)

    def map(self, func, na_action=None, **kwargs):
        return self._backend.map(func=func, na_action=na_action)

    def unique(self):
        return self._backend.unique()

    def isin(self, values: Iterable) -> Iterable[bool]:
        raise NotImplementedError

    def groupby(self):
        raise NotImplementedError

    @property
    def index(self):
        return self._backend.index

    @property
    def columns(self):
        return self._backend.columns


class ExperimentData(Dataset):
    def __init__(self, data: Any):
        if isinstance(data, Dataset):
            self.additional_fields = Dataset(data.data)._create_empty(
                data.index, data.columns
            )
            self.stats_fields = Dataset(backend)._create_empty(data.index, data.columns)
            self.analysis_tables = {}
        else:
            self.additional_fields = Dataset(data)
            self.stats_fields = Dataset(data)
            self.analysis_tables = {}

    def _create_empty(self, indexes=None, columns=None):
        self.additional_fields._create_empty(indexes, columns)
        self.stats_fields._create_empty(indexes, columns)
        return self

    # TODO переделать: обновление данных + обновление ролей
    def add_to_additional_fields(self, data: pd.DataFrame):
        self.additional_fields.data = self.additional_fields.data.join(data, how="left")

    # TODO переделать: обновление данных + обновление ролей
    def add_to_stats_fields(self, data: pd.DataFrame):
        self.stats_fields = self.stats_fields.data.join(data, how="left")

    def add_to_analysis_tables(
        self,
        key: str,
        data: pd.DataFrame,
        roles: Optional[Dict[ABCRole, Union[List[str], str]]] = None,
    ):
        self.analysis_tables[key] = Dataset(data, roles)
