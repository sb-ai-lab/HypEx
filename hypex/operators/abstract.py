from abc import abstractmethod, ABC
from typing import Optional, Any

from hypex.dataset import ExperimentData, Dataset, Arg1Role, Arg2Role, StatisticRole
from hypex.experiments.base import Executor
from hypex.utils import ExperimentDataEnum


class BinaryOperator(Executor, ABC):

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
