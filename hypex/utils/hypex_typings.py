import datetime
from typing import Union, Any

import numpy as np

StratificationRoleTypes = Union[int, str, datetime.datetime]
TreatmentRoleTypes = Union[int, bool, str]
TargetRoleTypes = PreTargetRoleTypes = Union[np.float64, int, bool]
ABCRoleTypes = Union[Any]
