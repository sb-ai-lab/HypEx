
rom abc import ABC, abstractmethod

import numpy as np
from scipy.stats import mode

from hypex.experiment.base import Executor


class StatDescriptive(Executor):
    statistic: float
    def __init__(self, field: str, descriptive_func=None):
        self.field = field
        self.descriptive_func = descriptive_func
        self.kwargs = {}
    
    @abstractmethod
    def execute(self, data):
        self.statistic = self.descriptive_func(data[self.field])

class StatMean(StatDescriptive):
    def __init__(self, field: str):
        super().__init__(field, np.mean)

class StatMedian(StatDescriptive):
    def __init__(self, field: str):
        super().__init__(field, np.median)

class StatMode(StatDescriptive):
    def __init__(self, field: str):
        super().__init__(field, mode)

class StatStd(StatDescriptive):
    def __init__(self, field: str, ddof=0):
        super().__init__(field, np.std, ddof=ddof)

class StatVariance(StatDescriptive):
    def __init__(self, field: str, ddof=0):
        super().__init__(field, np.var, ddof=ddof)

class StatMin(StatDescriptive):
    def __init__(self, field: str):
        super().__init__(field, np.min)

class StatMax(StatDescriptive):
    def __init__(self, field: str):
        super().__init__(field, np.max)

class StatRange(StatDescriptive):
    def __init__(self, field: str):
        super().__init__(field, np.ptp)

class StatSize(StatDescriptive):
    def __init__(self, field: str):
        super().__init__(field, len)
