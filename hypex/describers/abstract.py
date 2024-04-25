from abc import ABC
from typing import Optional, Any, Union

from hypex.dataset import ExperimentData, Dataset
from hypex.experiments.base import Executor
from hypex.utils import ExperimentDataEnum, FieldKey


class Describer(Executor, ABC):

    def __init__(
        self, target_field: FieldKey, full_name: Optional[str] = None, key: Any = ""
    ):
        self.target_field = target_field
        super().__init__(full_name, key)

    def _set_value(
        self, data: ExperimentData, value: Union[Dataset, None] = None, key=None
    ) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.analysis_tables, self._id, str(self.full_name), value
        )
