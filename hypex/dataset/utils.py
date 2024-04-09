from typing import Dict, Union

from hypex.dataset.roles import default_roles, ABCRole


def parse_roles(roles: Dict) -> Dict[Union[str, int], ABCRole]:
    new_roles = {}
    roles = roles or {}
    for role in roles:
        r = default_roles.get(role, role)
        if isinstance(roles[role], list):
            for i in roles[role]:
                new_roles[i] = r
        else:
            new_roles[roles[role]] = r
    return new_roles or roles
