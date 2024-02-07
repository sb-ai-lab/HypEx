from abc import ABC, abstractmethod

import numpy as np

from hypex.experiment.base import Executor

class MetricComparator(ABC, Executor):
    def __init__(self, x1_field: str, x2_field: str, out_field: str):
        self.x1_field = x1_field
        self.x2_field = x2_field
        self.out_field = out_field

    @abstractmethod
    def calc(x1, x2):
        pass

    def apply(self, data):
        data[self.out_field] = data.apply(
            lambda row: self.calc(row[self.x1_field], row[self.x2_field]), axis=1
        )
        return data

    def execute(self, data):
        data[self.out_field] = self.calc(data[self.x1_field], data[self.x2_field])
        return data


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
