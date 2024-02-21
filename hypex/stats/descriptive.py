from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from scipy.stats import mode

from hypex.experiment.base import Executor
from hypex.dataset.dataset import ExperimentData


class StatDescriptive(Executor):
    def __init__(self, field, descriptive_func: Callable = None, **kwargs):
        self.field = field
        self.descriptive_func = descriptive_func
        self.kwargs = kwargs

    def _set_value(self, data: ExperimentData, value) -> ExperimentData:
        data.set_value(
            "stats_fields", self._id, self.get_full_name(), value, key=self.field
        )
        return data

    @abstractmethod
    def execute(self, data: ExperimentData) -> ExperimentData:
        return self._set_value(
            data, self.descriptive_func(data[self.field], **self.kwargs)
        )


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
