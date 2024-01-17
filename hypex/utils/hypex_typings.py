from typing import Union, Any
import numpy as np
import datetime


StratificationRoleTypes = Union[int, str, datetime.datetime]
TreatmentRoleTypes = Union[int, bool, str]
TargetRoleTypes = PretargetRoleTypes = Union[np.float64, int, bool]
FeatureRoleTypes = ABCRoleTypes = InfoRoleTypes = Union[Any]