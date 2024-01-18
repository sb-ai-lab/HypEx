from abc import ABC
from typing import Any, Dict

import pandas

from roles import ABCRole
from hypex.hypotheses.base import Hypothesis


class ABCDataset(ABC):
    def __init__(self,
                 data: Any = None,
                 roles: Dict[ABCRole, list[str]] = None,
                 task: Hypothesis = None):
        self.task = task
        if data is not None:
            self.set_data(data, roles)

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

