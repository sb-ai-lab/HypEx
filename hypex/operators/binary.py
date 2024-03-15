from abc import ABC, abstractmethod

import numpy as np

from hypex.experiment.experiment import Executor
from hypex.dataset.dataset import ExperimentData

# TODO: replace x1_field and x2_field on subroles
class BinaryOperator(ABC, Executor):
    def get_full_name(self):
        return f"{self.__class__.__name__}({self.x1_field}, {self.x2_field})"

    def __init__(self, x1_field, x2_field, full_name: str = None, index: int = 0):
        self.x1_field = x1_field
        self.x2_field = x2_field
        super().__init__(full_name, index)

    def _set_value(self, data: ExperimentData, value) -> ExperimentData:
        data.set_value("additional_fields", self._id, self.get_full_name(), value)
        return data

    @staticmethod
    @abstractmethod
    def calc(x1, x2):
        raise NotImplementedError

    def apply(self, data: ExperimentData) -> ExperimentData:
        return self._set_value(
            data,
            data.apply(
                lambda row: self.calc(row[self.x1_field], row[self.x2_field]), axis=1
            ),
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        return self._set_value(
            data, self.calc(data[self.x1_field], data[self.x2_field])
        )


class MetricDelta(MetricComparator):
    @staticmethod
    def calc(x1, x2):
        return x2 - x1


class MetricPercentageDelta(MetricComparator):
    @staticmethod
    def calc(x1, x2):
        return (1 - x1 / x2) * 100


class MetricAbsoluteDelta(MetricComparator):
    @staticmethod
    def calc(x1, x2):
        return np.abs(x2 - x1)


class MetricRelativeDelta(MetricComparator):
    @staticmethod
    def calc(x1, x2):
        return 1 - x1 / x2


class MetricRatio(MetricComparator):
    @staticmethod
    def calc(x1, x2):
        return x1 / x2


class MetricLogRatio(MetricComparator):
    @staticmethod
    def calc(x1, x2):
        return np.log(x1 / x2)


class MetricPercentageRatio(MetricComparator):
    @staticmethod
    def calc(x1, x2):
        return (1 - x1 / x2) * 100
