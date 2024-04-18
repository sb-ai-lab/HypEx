from abc import ABC
from typing import Optional

from hypex.utils import (
    ABCRoleTypes,
    StratificationRoleTypes,
    TargetRoleTypes,
    TreatmentRoleTypes,
)


class ABCRole(ABC):
    _role_name = "Abstract"

    def __init__(self, data_type: Optional[ABCRoleTypes] = None):
        self.data_type = data_type

    @property
    def role_name(self) -> str:
        return self._role_name

    def __repr__(self) -> str:
        return self._role_name


class InfoRole(ABCRole):
    _role_name = "Info"


class StratificationRole(ABCRole):
    _role_name = "Stratification"

    def __init__(self, data_type: Optional[StratificationRoleTypes] = None):
        super().__init__(data_type)


class GroupingRole(ABCRole):
    _role_name = "Grouping"


class TreatmentRole(ABCRole):
    _role_name = "Treatment"

    def __init__(self, data_type: Optional[TreatmentRoleTypes] = None):
        super().__init__(data_type)


class TmpTreatmentRole(ABCRole):
    _role_name = "TmpTreatment"

    def __init__(self, data_type: Optional[TreatmentRoleTypes] = None):
        super().__init__(data_type)


class TargetRole(ABCRole):
    _role_name = "Target"

    def __init__(self, data_type: Optional[TargetRoleTypes] = None):
        super().__init__(data_type)


class FeatureRole(ABCRole):
    _role_name = "Feature"


class PreTargetRole(TargetRole):
    _role_name = "PreTarget"


class TempTargetRole(ABCRole):
    _role_name = "TempTarget"


class TempGroupingRole(ABCRole):
    _role_name = "TempGrouping"


class Arg1Role(ABCRole):
    _role_name = "Arg1"


class Arg2Role(ABCRole):
    _role_name = "Arg2"


class StatisticRole(ABCRole):
    _role_name = "Statistic"


default_roles = {
    "info": InfoRole(),
    "feature": FeatureRole(),
    "treatment": TreatmentRole(),
    "grouping": GroupingRole(),
    "target": TargetRole(),
    "pretarget": PreTargetRole(),
    "stratification": StratificationRole(),
    "statistic": StatisticRole(),
}
