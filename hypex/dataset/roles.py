from __future__ import annotations

from abc import ABC
from copy import deepcopy

from ..utils import (
    CategoricalTypes,
    DefaultRoleTypes,
    FeatureRoleTypes,
    RoleNameType,
    TargetRoleTypes,
)


class ABCRole(ABC):
    _role_name: RoleNameType = "Abstract"

    def __init__(self, data_type: DefaultRoleTypes | None = None):
        self.data_type = data_type

    @property
    def role_name(self) -> str:
        return self._role_name

    def __repr__(self) -> str:
        return f"{self._role_name}({self.data_type})"

    def astype(self, data_type: DefaultRoleTypes | None = None) -> ABCRole:
        role = deepcopy(self)
        role.data_type = data_type
        return role

    def asadditional(self, data_type: DefaultRoleTypes | None = None) -> ABCRole:
        data_type = data_type or self.data_type
        for role_type in list(default_roles.values()):
            if isinstance(role_type, self.__class__) and isinstance(
                role_type, AdditionalRole
            ):
                return role_type.__class__(data_type)
        return self.__class__(data_type)


class LagRole(ABCRole):
    """Base class for roles that support temporal metadata (parent, lag)."""

    def __init__(
        self,
        data_type: DefaultRoleTypes | None = None,
        parent: str | None = None,
        lag: int | None = None,
    ):
        super().__init__(data_type)
        self.parent = parent
        self.lag = lag

    def __repr__(self) -> str:
        parts = []
        if self.data_type is not None:
            parts.append(f"data_type={self.data_type}")
        if self.parent is not None:
            parts.append(f"parent='{self.parent}'")
        if self.lag is not None:
            parts.append(f"lag={self.lag}")
        return (
            f"{self._role_name}({', '.join(parts)})"
            if parts
            else f"{self._role_name}()"
        )


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

    def __init__(
        self,
        data_type: TargetRoleTypes | None = None,
        cofounders: list[str] | None = None,
    ):
        super().__init__(data_type=data_type)
        self.cofounders = cofounders if cofounders is not None else []


class FeatureRole(LagRole):
    _role_name: RoleNameType = "Feature"

    def __init__(
        self,
        data_type: FeatureRoleTypes | None = None,
        parent: str | None = None,
        lag: int | None = None,
    ):
        super().__init__(data_type=data_type, parent=parent, lag=lag)


class PreTargetRole(LagRole):
    _role_name: RoleNameType = "PreTarget"

    def __init__(
        self,
        data_type: TargetRoleTypes | None = None,
        parent: str | None = None,
        lag: int | None = None,
        cofounders: list[str] | None = None,
    ):
        super().__init__(data_type=data_type, parent=parent, lag=lag)
        self.cofounders = cofounders if cofounders is not None else []


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


class AdditionalTreatmentRole(AdditionalRole, TreatmentRole):
    _role_name: RoleNameType = "AdditionalTreatment"


class AdditionalGroupingRole(AdditionalRole, GroupingRole):
    _role_name: RoleNameType = "AdditionalGrouping"


class AdditionalTargetRole(AdditionalRole, TargetRole):
    _role_name: RoleNameType = "AdditionalTarget"


class AdditionalFeatureRole(AdditionalRole, FeatureRole):
    _role_name: RoleNameType = "AdditionalTarget"


class AdditionalPreTargetRole(AdditionalRole, PreTargetRole):
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
    "additionalfeature": AdditionalFeatureRole(),
    "additionalpretarget": AdditionalPreTargetRole(),
}
