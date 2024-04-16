from typing import Union, Dict, List

import numpy as np

from hypex.dataset.dataset import Dataset, ExperimentData
from hypex.dataset.roles import StatisticRole
from hypex.describers.base import Describer


class Unique(Describer):

    def calc(self, data: Dataset):
        return self._convert_to_dataset(
            [{self.full_name: np.unique(data.data[self.target_field])}]
        )

    def _convert_to_dataset(self, data: Union[List, Dict]) -> Dataset:
        return Dataset.from_dict(data, {self.id: StatisticRole()})

    def execute(self, data: ExperimentData) -> ExperimentData:
        result_dataset = self.calc(data)
        return self._set_value(data, result_dataset)
