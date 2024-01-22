from typing import Optional, Dict, Union
from pandas import DataFrame
from hypex.dataset.roles import ABCRole
from hypex.errors.roleserrors import *
from hypex.hypotheses.base import Hypothesis
from hypex.dataset.base import DatasetSeletor, PandasDataset


class Dataset:
    class Locker:
        def __init__(self, df: DataFrame, method: str):
            self.df = df
            self.method = getattr(df, method)

        def __getitem__(self, item):
            return self.method[item]

    def __init__(self,
                 data: Optional[DataFrame] = None,
                 roles: Optional[Dict[ABCRole, Union[list[str], str]]] = None,
                 task: Optional[Hypothesis] = Hypothesis('auto')
                 ):
        if data is not None:
            self.set_data(data, task, roles)

    def set_data(self, data: DataFrame, task, roles):
        self.roles = roles or {}
        self.task = task
        if self._check(self.task, self.roles):
            self.data = DatasetSeletor().select_dataset(data)
            self._set_extra_attributes()

    @staticmethod
    def _check(task: Hypothesis,
              roles: Union[Dict[ABCRole, Union[list[str], str]], None]):
        for role in roles:
            if role not in task.attributes_for_test:
                raise RoleError
        return 1

    def _set_extra_attributes(self):
        if isinstance(self.data, PandasDataset):
            self.loc = self.Locker(self.data.data, 'loc')
            self.iloc = self.Locker(self.data.data, 'iloc')

    def __repr__(self):
        return self.data.__repr__()

    def apply(self, func, axis=0, raw=False,
              result_type=None, args=(), by_row='compat', **kwargs):
        try:
            return self.data.apply(func, axis, raw, result_type, args, by_row, **kwargs)
        except AttributeError:
            raise MethodError('apply', self.data.__class__.__name__)

    def map(self, func, na_action=None, **kwargs):
        try:
            return self.data.map(func, na_action, **kwargs)
        except AttributeError:
            raise MethodError('map', self.data.__class__.__name__)
