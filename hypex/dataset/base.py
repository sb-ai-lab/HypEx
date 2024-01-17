from abc import ABC
from typing import Any, Dict
from roles import ABCRole


class Hypothesis:
    pass


class ABCDataset(ABC):
    # _check_columns = (наличие столбцов для задачи)
    def __init__(self,
                 data: Any,
                 roles: Dict[ABCRole, list[str]],
                 task: Hypothesis):
        if data and self.check(roles, task):
            self.set_data(data, roles)

    def check(self,
              roles: Dict[ABCRole, list[str]],
              task: Hypothesis):
        pass # наличие столбцов, корректные типы и тд

    def set_data(self,
                 data: Any,
                 roles: Dict[ABCRole, list[str]]):
        self.data = data
        self.roles = roles


class ABCColumn(ABC):
    def __init__(self,
                 data: Any,
                 role: ABCRole):
        self.data = data
        self.role = role


