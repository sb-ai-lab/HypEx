from __future__ import annotations

from abc import ABC

from ..utils import CategoricalTypes, DefaultRoleTypes, RoleNameType, TargetRoleTypes


class ABCRole(ABC):
    _role_name: RoleNameType = "Abstract"

    def __init__(self, data_type: DefaultRoleTypes | None = None):
        self.data_type = data_type

    @property
    def role_name(self) -> str:
        return self._role_name

    def __repr__(self) -> str:
        return f"{self._role_name}({self.data_type})"


class InfoRole(ABCRole):
    _role_name: RoleNameType = "Info"


class StratificationRole(ABCRole):
    _role_name: RoleNameType = "Stratification"

    def __init__(self, data_type: CategoricalTypes | None = None):
        super().__init__(data_type)


class GroupingRole(ABCRole):
    _role_name: RoleNameType = "Grouping"

    def __init__(self, data_type: CategoricalTypes | None = None):
        super().__init__(data_type)


class TreatmentRole(ABCRole):
    _role_name: RoleNameType = "Treatment"


class TargetRole(ABCRole):
    _role_name: RoleNameType = "Target"

    def __init__(self, data_type: TargetRoleTypes | None = None):
        super().__init__(data_type)


class FeatureRole(ABCRole):
    _role_name: RoleNameType = "Feature"


class PreTargetRole(ABCRole):
    _role_name: RoleNameType = "PreTarget"

    def __init__(self, data_type: TargetRoleTypes | None = None):
        super().__init__(data_type)


class StatisticRole(ABCRole):
    _role_name: RoleNameType = "Statistic"


class ResumeRole(ABCRole):
    _role_name = "Resume"


class FilterRole(ABCRole):
    _role_name: RoleNameType = "Filter"


class ConstGroupRole(ABCRole):
    _role_name: RoleNameType = "ConstGroup"


# ___________________________________________________________________________________________
class TempRole(ABCRole):
    _role_name: RoleNameType = "Temp"


class TempTreatmentRole(TempRole, TreatmentRole):
    _role_name: RoleNameType = "TempTreatment"


class TempTargetRole(TempRole, TargetRole):
    _role_name: RoleNameType = "TempTarget"


class TempGroupingRole(TempRole, GroupingRole):
    _role_name: RoleNameType = "TempGrouping"


class DefaultRole(ABCRole):
    _role_name: RoleNameType = "Default"


class ReportRole(ABCRole):
    _role_name: RoleNameType = "Report"


# ___________________________________________________________________________________________
class AdditionalRole(ABCRole):
    _role_name: RoleNameType = "Additional"


class AdditionalTreatmentRole(AdditionalRole):
    _role_name: RoleNameType = "AdditionalTreatment"


class AdditionalGroupingRole(AdditionalRole):
    _role_name: RoleNameType = "AdditionalGrouping"


class AdditionalTargetRole(AdditionalRole):
    _role_name: RoleNameType = "AdditionalTarget"


class AdditionalPreTargetRole(AdditionalRole):
    _role_name: RoleNameType = "AdditionalPreTarget"


class AdditionalMatchingRole(AdditionalRole):
    _role_name: RoleNameType = "AdditionalMatching"


default_roles: dict[RoleNameType, ABCRole] = {
    "info": InfoRole(),
    "default": DefaultRole(),
    "feature": FeatureRole(),
    "treatment": TreatmentRole(),
    "grouping": GroupingRole(),
    "target": TargetRole(),
    "pretarget": PreTargetRole(),
    "stratification": StratificationRole(),
    "statistic": StatisticRole(),
    "filter": FilterRole(),
    "constgroup": ConstGroupRole(),
    "additionaltreatment": AdditionalTreatmentRole(),
    "additionalgrouping": AdditionalGroupingRole(),
    "additionaltarget": AdditionalTargetRole(),
    "additionalpretarget": AdditionalPreTargetRole(),
}
