from abc import ABC
from typing import Union

from hypex.utils.hypex_typings import (
    ABCRoleTypes,
    StratificationRoleTypes,
    TargetRoleTypes,
    TreatmentRoleTypes,
)


class ABCRole(ABC):
    _role_name = "Abstract"

    def __init__(self, data_type: Union[ABCRoleTypes, None] = None):
        self.data_type = data_type


class InfoRole(ABCRole):
    _role_name = "Info"


class StratificationRole(ABCRole):
    _role_name = "Stratification"

    def __init__(self, data_type: Union[StratificationRoleTypes, None] = None):
        super().__init__(data_type)


class GroupingRole(ABCRole):
    _role_name = "Grouping"


class TreatmentRole(ABCRole):
    _role_name = "Treatment"

    def __init__(self, data_type: Union[TreatmentRoleTypes, None] = None):
        super().__init__(data_type)


class TargetRole(ABCRole):
    _role_name = "Target"

    def __init__(self, data_type: Union[TargetRoleTypes, None] = None):
        super().__init__(data_type)


class FeatureRole(ABCRole):
    _role_name = "Feature"


class PreTargetRole(TargetRole):
    _role_name = "PreTarget"


class TempTargetRole(TargetRole):
    _role_name = "TempTarget"


class TempGroupingRole(GroupingRole):
    _role_name = "TempGrouping"


class Arg1Role:
    _role_name = "Arg1"


class Arg2Role:
    _role_name = "Arg2"


class StatisticRole(ABCRole):
    _role_name = "Statistic"


default_roles = {
    "info": InfoRole,
    "feature": FeatureRole,
    "treatment": TreatmentRole,
    "grouping": GroupingRole,
    "target": TargetRole,
    "pretarget": PreTargetRole,
    "stratification": StratificationRole,
    "statistic": StatisticRole,
}
