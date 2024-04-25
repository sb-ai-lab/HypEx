from typing import Optional

import numpy as np

from hypex.dataset import Dataset
from hypex.operators.abstract import BinaryOperator


class MetricDelta(BinaryOperator):
    def calc(self, data: Dataset, other: Optional[Dataset] = None):
        if other is not None:
            return data.data - other.data


class MetricPercentageDelta(BinaryOperator):
    def calc(self, data: Dataset, other: Optional[Dataset] = None):
        if other is not None:
            return (1 - data.data / other.data) * 100


class MetricAbsoluteDelta(BinaryOperator):
    def calc(self, data: Dataset, other: Optional[Dataset] = None):
        if other is not None:
            return np.abs(data.data - other.data)


class MetricRelativeDelta(BinaryOperator):
    def calc(self, data: Dataset, other: Optional[Dataset] = None):
        if other is not None:
            return 1 - data.data / other.data


class MetricRatio(BinaryOperator):
    def calc(self, data: Dataset, other: Optional[Dataset] = None):
        if other is not None:
            return data.data / other.data


class MetricLogRatio(BinaryOperator):
    def calc(self, data: Dataset, other: Optional[Dataset] = None):
        if other is not None:
            return np.log(data.data / other.data)


class MetricPercentageRatio(BinaryOperator):
    def calc(self, data: Dataset, other: Optional[Dataset] = None):
        if other is not None:
            return (1 - data.data / other.data) * 100
