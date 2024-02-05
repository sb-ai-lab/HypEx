from typing import Dict, Optional, Union

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
        **kwargs
    ):
        return self.data.apply(func, axis, raw, result_type, args, by_row, **kwargs)

    def map(self, func, na_action=None, **kwargs):
        return self.data.map(func, na_action, **kwargs)
