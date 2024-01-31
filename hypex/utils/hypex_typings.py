import datetime
from typing import Any, Union

import numpy as np

StratificationRoleTypes = Union[np.float64, str, datetime.datetime]
TreatmentRoleTypes = Union[np.float64, bool, str]
TargetRoleTypes = Union[np.float64, int, bool]
ABCRoleTypes = Union[Any]
