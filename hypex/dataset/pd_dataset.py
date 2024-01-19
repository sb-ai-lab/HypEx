from hypex.dataset.base import *
from pandas import DataFrame
from typing import Optional, Union, Sequence
from hypex.errors.roleserrors import RoleError
import pandas


class PandasDataset(ABCDataset):
    def __init__(
            self,
            data: Optional[DataFrame] = None,
            roles: Optional[Dict[ABCRole, Union[list[str], str]]] = None,
            task: Optional[Union[Hypothesis, None]] = None
    ):
        self.task: Union[Hypothesis, None] = task
        if data is not None and self.check(task, roles):
            self.set_data(data, roles)

    def set_data(self, data, roles):
        super().set_data(data, roles)

    @staticmethod
    def check(task, roles):
        if roles:
            for role in roles:
                if role not in task.attributes_for_test:
                    raise RoleError
        return 1

    def _get_column_index(self, column_name: Union[Sequence[str], str]) -> Union[int, Sequence[int]]:
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

    def __setitem__(self, key, value):
        self.data[key] = value
        return self.data


if __name__ == "__main__":
    d = {'col1': [1, 2], 'col2': [3, 4]}
    d2 = {'col1': [1, 2], 'col2': [3, 4]}
    df = pandas.DataFrame(data=d)
    df2 = pandas.DataFrame(data=d2)
    abc = PandasDataset(df)
    df['new_col'] = [5, 6]
    abc['new_col'] = [5, 6]
    print(abc[['col1']])