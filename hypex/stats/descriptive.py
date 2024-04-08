from abc import abstractmethod

import numpy as np
from scipy.stats import mode

from hypex.dataset.dataset import ExperimentData, Dataset
from hypex.dataset.roles import TempTargetRole
from hypex.experiments.base import Executor
from hypex.utils.enums import ExperimentDataEnum


class StatDescriptive(Executor):
    def __init__(self, full_name=None, key=0, **kwargs):
        self.kwargs = kwargs
        super().__init__(full_name, key)

    @abstractmethod
    def calc(self, data: Dataset):
        raise NotImplementedError

    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        data.set_value(
            ExperimentDataEnum.stats,
            self.id,
            str(self.full_name),
            value,
            key=key,
        )
        return data

    def execute(self, data: ExperimentData) -> ExperimentData:
        target = data.get_columns_by_roles(TempTargetRole(), tmp_role=True)[0]
        return self._set_value(data, self.calc(data[target]), target)


class Mean(StatDescriptive):
    def calc(self, data):
        return data.data.mean()


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


class Size(StatDescriptive):
    def calc(self, data):
        return len(data)
