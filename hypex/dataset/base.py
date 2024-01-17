import typing
from abc import ABC
from typing import Any, Optional, Dict
from roles import ABCRole


class Hypothesis:
    pass


class ABCDataset(ABC):
    # _check_data = (_check_treatment(), _check_numeric())
    def __init__(self,
                 data: Any,
                 roles: Optional[Dict[ABCRole, list[str]]],
                 task: Optional[Hypothesis]):
        if data and self.check_roles(roles, task):
            self.set_data(data, roles)

    def check_roles(self,
                    roles: Dict[ABCRole, list[str]],
                    task: Optional[Hypothesis]):
        pass
    def set_data(self,
                 data: Any,
                 roles: Optional[Dict[ABCRole, list[str]]]):
        self.data = data
        self.roles = roles

class ABCColumn(ABC):
    def __init__(self,
                 data: Any,
                 role: Optional[ABCRole]):
        pass


