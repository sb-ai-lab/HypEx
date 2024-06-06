from typing import Union, Optional

from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu  # type: ignore

from .abstract import GroupComparator
from ..dataset import Dataset, ABCRole
from ..extensions.hypothesis_testing import (
    TTestExtension,
    KSTestExtension,
    UTestExtension,
)
from ..utils import SpaceEnum
from ..utils.typings import NumberTypes, CategoricalTypes 

class ATC(GroupComparator): 
    def _inner_function(data: Dataset, other: Optional[Dataset] = None, **kwargs) -> float: 
        other = other.iloc[other.search_columns(MatchRole())]
        return (data - other).mean()


class ATT(GroupComparator): 
    def _inner_function(data: Dataset, other: Optional[Dataset] = None, **kwargs) -> float: 
        data = data.iloc[data.search_columns(MatchRole())]
        return (data - other).mean() 


class ATE(GroupComparator): 
    def _inner_function(data: Dataset, other: Optional[Dataset] = None, **kwargs) -> float:
        atc = ATC.calc(data, other) 
        att = ATT.calc(data, other) 
        return (att * len(data) + atc * len(other)) / (len(data) + len(other))