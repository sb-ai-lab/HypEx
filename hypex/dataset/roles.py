import sys
from abc import ABC
from typing import Any
import numpy as np


class ABCRole(ABC):
    _role_name = 'Abstract'

    def get_from_str(self,
                     role_name: str, **kwargs: Any) -> "ABCRole":
        possible_roles = ['info', 'feature', 'stratification', 'grouping', 'treatment', 'target', 'pretarget']
        role_name = role_name.lower()
        if role_name in possible_roles:
            return getattr(sys.modules[__name__], role_name.title() + "Role")()
        raise ValueError(f"Unknown role: {role_name}. "
                         f"Possible roles: " + ("{} " * len(possible_roles)).format(*possible_roles))


class InfoRole(ABCRole):
    _role_name = 'Info'

    # нужен ли дефолтный тип данных?
    def __init__(self):
        pass


class StratificationRole(ABCRole):
    _role_name = 'Stratification'

    # нужен ли дефолтный тип данных?
    def __init__(self, data_type: type = int):
        self.data_type = data_type


class GroupingRole(ABCRole):
    _role_name = 'Grouping'


class TreatmentRole(ABCRole):
    _role_name = 'Treatment'

    def __init__(self, data_type: type = int):
        self.data_type = data_type

class TargetRole(ABCRole):
    _role_name = 'Target'

    def __init__(self, data_type: type = np.float64):
        self.data_type = data_type


class FeatureRole(ABCRole):
    _role_name = 'Feature'


class PreTargetRole(TargetRole):
    _role_name = 'PreTarget'
