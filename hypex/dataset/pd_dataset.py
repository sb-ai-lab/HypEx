from hypex.dataset.base import *
from pandas import DataFrame, Series
from typing import Optional, Union, Sequence


class PandasDataset(ABCDataset):
    def __init__(
            self,
            data: Optional[DataFrame] = None,
            roles: Optional[Dict[ABCRole, Union[list[str], str]]] = None,
            task: Optional[Union[Hypothesis, None]] = None
    ):
        self.task = task
        if data is not None and self.check(task, roles):
            self.set_data(data, roles)

    def set_data(self, data, roles):
        super().set_data(data, roles)

    def check(self, task, roles):
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
        raise KeyError("Not implemented yet")


if __name__ == "__main__":
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pandas.DataFrame(data=d)
    abc = PandasDataset(df)
    print(abc[['col3', 'col2']])