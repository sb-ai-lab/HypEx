from typing import List

from hypex.experiment.base import Experiment, ExperimentMulti


class SplitterAA(Experiment):
    test_indexes: List
    control_indexes: List

    def __init__(
        self,
        test_size: float = 0.5,
        significance: float = 0.05,
        random_state: int = None,
    ):
        self.test_size = test_size
        self.random_state = random_state


class SplitterAASimple(SplitterAA):
    def execute(self, data):
        addition_indexes = list(shuffle(data.index, random_state=self.random_state))
        edge = int(len(addition_indexes) * test_size)
        self.test_indexes = addition_indexes[:edge]
        self.control_indexes = addition_indexes[edge:]


class SplitterAASimpleWithGrouping(SplitterAA):
    def execute(self, data):
        group_field = None
        random_ids = shuffle(data[group_field].unique(), random_state=self.random_state)
        edge = int(len(random_ids) * self.test_size)
        result["test_indexes"] = list(
            data[data[group_field].isin(random_ids[:edge])].index
        )
        result["control_indexes"] = list(
            data[data[group_field].isin(random_ids[edge:])].index
        )


class SplitterAAWithStratification(SplitterAA):
    def __init__(
        self,
        inner_splitter: SplitterAA,
        test_size: float = 0.5,
        significance: float = 0.05,
        random_state: int = None,
    ):
        super().__init__(test_size, significance, random_state)
        self.inner_splitter = inner_splitter

    def execute(self, data):
        self.control_indexes = []
        self.test_indexes = []
        stratification_columns = None

        groups = data.groupby(stratification_columns)
        for _, gd in groups:
            self.inner_splitter.execute(data)
            self.control_indexes.extend(self.inner_splitter.control_indexes)
            self.test_indexes.extend(self.inner_splitter.test_indexes)

class SplitterAAMulti(ExperimentMulti):
    def execute(self, data):
        pass