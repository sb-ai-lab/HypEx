from abc import abstractmethod

from hypex.dataset.dataset import ExperimentData, Dataset
from hypex.dataset.roles import TempTargetRole
from hypex.experiments.base import Executor
from hypex.utils.enums import ExperimentDataEnum


class AggStat(Executor):
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
