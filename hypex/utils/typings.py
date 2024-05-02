import datetime
from typing import Any, Union, Dict, List

StratificationRoleTypes = Union[float, str, datetime.datetime]
DefaultRoleTypes = Union[float, bool, str, int]
TargetRoleTypes = Union[float, int, bool]
CategoricalTypes = str

FieldKey = Union[str, int]

FromDictType = Union[Dict[str, List[Any]], List[Dict[Any, Any]]]
RoleNameType = str
