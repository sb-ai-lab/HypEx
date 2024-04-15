from abc import abstractmethod
from typing import Any, Optional

import numpy as np

from hypex.dataset.dataset import ExperimentData, Dataset
from hypex.dataset.roles import Arg1Role, Arg2Role, StatisticRole
from hypex.experiments.base import Executor
from hypex.utils.enums import ExperimentDataEnum


class BinaryOperator(Executor):

    def __init__(self, full_name: Optional[str] = None, key: Any = ""):
        super().__init__(full_name, key)

    def _set_value(
        self, data: ExperimentData, value: Any, key: Any = None
    ) -> ExperimentData:
        data.set_value(
            ExperimentDataEnum.additional_fields,
            self._id,
            str(self.full_name),
            value,
            role=StatisticRole(),
        )
        return data

    @abstractmethod
    def calc(self, data: Dataset, other: Optional[Dataset] = None):
        raise NotImplementedError

    def execute(self, data: ExperimentData) -> ExperimentData:
        arg1 = data.get_columns_by_roles(Arg1Role(), tmp_role=True)[0]
        arg2 = data.get_columns_by_roles(Arg2Role(), tmp_role=True)[0]
        return self._set_value(data, self.calc(data[arg1], data[arg2]))


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
