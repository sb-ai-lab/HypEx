import sys
from abc import ABC
from hypex.utils.hypex_typings import *


MatchingRoles = []
AARoles = []
ABRoles = []

class ABCRole(ABC):
    _role_name = 'Abstract'

    def __init__(self, data_type: ABCRoleTypes = Any):
        self.data_type = data_type

    def get_from_str(self,
                     role_name: str, **kwargs: Any) -> "ABCRole":
        possible_roles = ['info', 'feature', 'stratification', 'grouping', 'treatment', 'target', 'pretarget']
        role_name = role_name.lower()
        if role_name in possible_roles:
            return getattr(sys.modules[__name__], role_name.title() + "Role")(
                kwargs.get('data_type', getattr(sys.modules[__name__], role_name.title() + "RoleTypes"))
            )
        raise ValueError(f"Unknown role: {role_name}. "
                         f"Possible roles: " + ("{} " * len(possible_roles)).format(*possible_roles))


class InfoRole(ABCRole):
    _role_name = 'Info'

    def __init__(self, data_type: InfoRoleTypes):
        super().__init__(data_type)


class StratificationRole(ABCRole):
    _role_name = 'Stratification'

    def __init__(self, data_type: StratificationRoleTypes):
        super().__init__(data_type)


class GroupingRole(ABCRole):
    _role_name = 'Grouping'


class TreatmentRole(ABCRole):
    _role_name = 'Treatment'

    def __init__(self, data_type: TreatmentRoleTypes):
        super().__init__(data_type)


class TargetRole(ABCRole):
    _role_name = 'Target'

    def __init__(self, data_type: TargetRoleTypes):
        super().__init__(data_type)


class FeatureRole(ABCRole):
    _role_name = 'Feature'

    def __init__(self, data_type: FeatureRoleTypes):
        super().__init__(data_type)


class PretargetRole(TargetRole):
    _role_name = 'PreTarget'


class DropRole(ABCRole):
    _role_name = 'DropRole'

