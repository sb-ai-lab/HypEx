import datetime
from typing import Any, Union, Dict, List

from hypex.dataset.roles import ABCRole

StratificationRoleTypes = Union[float, str, datetime.datetime]
TreatmentRoleTypes = Union[float, bool, str]
TargetRoleTypes = Union[float, int, bool]
ABCRoleTypes = Union[Any]

FieldKey = Union[str, int]

RolesType = Union[Dict[ABCRole, Union[List[str], str]], Dict[str, ABCRole]]
FromDictType = Union[Dict[str, List[Any]], List[Dict[Any, Any]]]
