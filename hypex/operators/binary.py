from abc import abstractmethod
from typing import Any

import numpy as np

from hypex.dataset.dataset import ExperimentData
from hypex.dataset.roles import Arg1Role, Arg2Role
from hypex.experiment.experiment import Executor
from hypex.utils.enums import ExperimentDataEnum


class BinaryOperator(Executor):
    def get_full_name(self):
        return f"{self.__class__.__name__}"

    def __init__(self, full_name: str = None, key: Any = 0):
        super().__init__(full_name, key)

    def _set_value(self, data: ExperimentData, value) -> ExperimentData:
        data.set_value(
            ExperimentDataEnum.additional_fields, self._id, self.get_full_name(), value
        )
        return data

    @staticmethod
    @abstractmethod
    def calc(x1, x2):
        raise NotImplementedError

    def apply(self, data: ExperimentData) -> ExperimentData:
        arg1 = data.data.get_columns_by_roles(Arg1Role, tmp_role=True)[0]
        arg2 = data.data.get_columns_by_roles(Arg2Role, tmp_role=True)[0]
        return self._set_value(
            data,
            data.apply(lambda row: self.calc(row[arg1], row[arg2]), axis=1),
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        arg1 = data.data.get_columns_by_roles(Arg1Role, tmp_role=True)[0]
        arg2 = data.data.get_columns_by_roles(Arg2Role, tmp_role=True)[0]
        return self._set_value(data, self.calc(data[arg1], data[arg2]))


class MetricDelta(BinaryOperator):
    @staticmethod
    def calc(x1, x2):
        return x2 - x1


class MetricPercentageDelta(BinaryOperator):
    @staticmethod
    def calc(x1, x2):
        return (1 - x1 / x2) * 100


class MetricAbsoluteDelta(BinaryOperator):
    @staticmethod
    def calc(x1, x2):
        return np.abs(x2 - x1)


class MetricRelativeDelta(BinaryOperator):
    @staticmethod
    def calc(x1, x2):
        return 1 - x1 / x2


class MetricRatio(BinaryOperator):
    @staticmethod
    def calc(x1, x2):
        return x1 / x2


class MetricLogRatio(BinaryOperator):
    @staticmethod
    def calc(x1, x2):
        return np.log(x1 / x2)


class MetricPercentageRatio(BinaryOperator):
    @staticmethod
    def calc(x1, x2):
        return (1 - x1 / x2) * 100
