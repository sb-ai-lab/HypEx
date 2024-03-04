from typing import List

from hypex.experiment.experiment import Executor, Experiment
from hypex.dataset.dataset import ExperimentData, Dataset
from hypex.dataset.roles import GroupingRole, StratificationRole
from hypex.transformers.transformers import Shuffle

# TODO: To Experiment
# TODO: Set group


class SplitterAA(Experiment):
    def __init__(
        self,
        test_size: float = 0.5,
        random_state: int = None,
        full_name: str = None,
        index: int = 0,
    ):
        self.test_size = test_size
        self.random_state = random_state
        super().__init__(
            [
                Shuffle(random_state=self.random_state),
            ],
            full_name,
            index,
        )

    @property
    def _is_transformer(self):
        return True

    def generate_params_hash(self) -> str:
        return f"{self.random_state}"

    def _set_value(self, data: ExperimentData, value) -> ExperimentData:
        return data.set_value("additional_fields", self._id, self.get_full_name(), value, role=GroupingRole)

    def execute(self, data: ExperimentData) -> ExperimentData:
        experiment_data: ExperimentData = super().execute(data)

        addition_indexes = list(experiment_data.index)
        edge = int(len(addition_indexes) * self.test_size)

        result_group = ['A' if i < edge else 'B' for i in addition_indexes]
        data = self._set_value(data, result_group)

        return data

# TODO: Unique is comparator?
class SplitterAAWithGrouping(SplitterAA):
    def execute(self, data: ExperimentData):
        group_field = data.get_columns_by_roles(GroupingRole)
        random_ids = shuffle(data[group_field].unique(), random_state=self.random_state)
        edge = int(len(random_ids) * self.test_size)

        data["treatment"] = 0
        test_indexes = list(data[data[group_field].isin(random_ids[:edge])].index)
        data["treatment"][test_indexes] = 1
        return data


class SplitterAAWithStratification(SplitterAA):
    def __init__(
        self,
        inner_splitter: SplitterAA,
        test_size: float = 0.5,
        random_state: int = None,
        full_name: str = None,
        index: int = 0,
    ):
        self.inner_splitter = inner_splitter
        super().__init__(test_size, random_state, full_name, index)

    def execute(self, data):
        control_indexes = []
        test_indexes = []
        stratification_columns = data.get_columns_by_roles(StratificationRole)

        groups = data.groupby(stratification_columns)
        for _, gd in groups:
            self.inner_splitter.execute(data)
            self.control_indexes.extend(self.inner_splitter.control_indexes)
            self.test_indexes.extend(self.inner_splitter.test_indexes)


class SplitterAAMulti(ExperimentMulti):
    def execute(self, data):
        raise NotImplementedError
