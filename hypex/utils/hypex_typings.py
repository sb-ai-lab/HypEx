import datetime
from typing import Any, Union

StratificationRoleTypes = Union[float, str, datetime.datetime]
TreatmentRoleTypes = Union[float, bool, str]
TargetRoleTypes = Union[float, int, bool]
ABCRoleTypes = Union[Any]
