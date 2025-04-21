from abc import ABC

from ..dataset import ExperimentData
from ..executor import Executor
from ..utils import ExperimentDataEnum


class Analyzer(Executor, ABC):
    """
    Abstract class for analyzers.
    """

    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.analysis_tables, self.id, value, key=key
        )
