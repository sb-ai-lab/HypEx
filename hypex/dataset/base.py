from abc import ABC
from typing import Any, Dict
from roles import ABCRole
from hypex.hypotheses.base import Hypothesis


class ABCDataset(ABC):
    def __init__(self,
                 data: Any,
                 roles: Dict[ABCRole, list[str]],
                 task: Hypothesis):
        self.task = task
        if data is not None:
            self.set_data(data, roles)

    def set_data(self,
                 data: Any,
                 roles: Dict[ABCRole, list[str]]):
        self.data = data
        self.roles = roles

    def __repr__(self):
        return self.data.__repr__()


class ABCColumn(ABC):
    def __init__(self,
                 data: Any,
                 role: ABCRole):
        self.data = data
        self.role = role
