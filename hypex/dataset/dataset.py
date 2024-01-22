from typing import Optional, Dict, Union

from pandas import DataFrame
from hypex.dataset.roles import ABCRole
from hypex.errors.roleserrors import RoleError
from hypex.hypotheses.base import Hypothesis
from hypex.dataset.base import DatasetSeletor, PandasDataset


class Dataset:
    class Locker:
        def __init__(self, df: DataFrame):
            self.df = df

        def __getitem__(self, item):
            return self.df.loc[item]

    class ILocker:
        def __init__(self, df: DataFrame):
            self.df = df

        def __getitem__(self, item):
            return self.df.iloc[item]

    def __init__(self,
                 data: Optional[DataFrame] = None,
                 roles: Optional[Dict[ABCRole, Union[list[str], str]]] = None,
                 task: Optional[Union[Hypothesis, None]] = None
                 ):
        if data is not None:
            self.set_data(data, task, roles)

    def set_data(self, data: DataFrame, task, roles):
        task = task if task is not None else Hypothesis('auto')
        roles = roles if roles is not None else {}
        if self.check(task, roles):
            self.data = DatasetSeletor().select_dataset(data)
            if isinstance(self.data, PandasDataset):
                self.loc = self.Locker(data)
                self.iloc = self.ILocker(data)

    @staticmethod
    def check(task: Union[Hypothesis, None],
              roles: Union[Dict[ABCRole, Union[list[str], str]], None]):
        for role in roles:
            if role not in task.attributes_for_test:
                raise RoleError
        return 1

    def __repr__(self):
        return self.data.__repr__()


if __name__ == "__main__":
    d = {'col1': [1, 2], 'col2': [3, 4]}
    d2 = {'col1': [1, 2], 'col2': [3, 4]}
    df = DataFrame(data=d)
    df2 = [1, 2, 3]
    abc = Dataset(df)
    abc2 = Dataset(df2)
    print(abc.loc[1])
    print(abc2.iloc[1])
