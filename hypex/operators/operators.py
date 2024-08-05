from copy import deepcopy
from typing import Optional, Any, List, Literal, Union, Dict

from hypex.dataset import Dataset, ABCRole, ExperimentData, TargetRole
from hypex.ml.faiss import FaissNearestNeighbors
from hypex.operators.abstract import GroupOperator
from hypex.utils.enums import ExperimentDataEnum



