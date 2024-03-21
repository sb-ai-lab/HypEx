from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from scipy.stats import mode

from hypex.experiment.base import Executor
from hypex.dataset.dataset import ExperimentData
from hypex.dataset.roles import TempTargetRole

class StatDescriptive(ABC, Executor):
    def __init__(self, full_name=None, index=0, **kwargs):
        self.kwargs = kwargs
        super().__init__(full_name, index)

    @abstractmethod
    def calc(self, data):
        raise NotImplementedError

    def _set_value(self, data: ExperimentData, value) -> ExperimentData:
        data.set_value(
            "stats_fields", self._id, self.get_full_name(), value, key=self.field
        )
        return data

    @abstractmethod
    def execute(self, data: ExperimentData) -> ExperimentData:
        target = data.data.get_columns_by_roles(TempTargetRole, tmp_role=True)[0]
        return self._set_value(
            data, self.calc(data[target])
        )


class Mean(StatDescriptive):
    def calc(self, data):
        return np.mean(data, **self.kwargs)

class Median(StatDescriptive):
    def calc(self, data):
        return np.median(data, **self.kwargs)

class Mode(StatDescriptive):
    def calc(self, data):
        return mode(data, **self.kwargs)


class Std(StatDescriptive):
    def calc(self, data):
        return np.std(data, **self.kwargs)


class Variance(StatDescriptive):
    def calc(self, data):
        return np.var(data, **self.kwargs)


class Min(StatDescriptive):
    def calc(self, data):
        return np.min(data, **self.kwargs)

class Max(StatDescriptive):
    def calc(self, data):
        return np.max(data, **self.kwargs)


class Range(StatDescriptive):
    def __init__(self, field: str):
        super().__init__(field, np.ptp)

class Size(StatDescriptive):
    def calc(self, data):
        return len(data, **self.kwargs)
