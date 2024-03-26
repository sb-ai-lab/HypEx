import datetime
from typing import Any, Union, Dict, List

StratificationRoleTypes = Union[float, str, datetime.datetime]
TreatmentRoleTypes = Union[float, bool, str]
TargetRoleTypes = Union[float, int, bool]
ABCRoleTypes = Union[Any]

FieldKey = Union[str, int]

FromDictType = Union[Dict[str, List[Any]], List[Dict[Any, Any]]]
