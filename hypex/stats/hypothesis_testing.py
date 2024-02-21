from abc import ABC
from

from scipy.stats import ttest_ind, ks_2samp

from hypex.experiment.base import Executor
from hypex.dataset.dataset import ExperimentData


class StatHypothesisTesting(ABC, Executor):
    def __init__(
        self,
        target_field: str,
        group_field: str,
        reliability: float = 0.05,
        power: float = 0.8,
    ):
        self.target_field = target_field
        self.group_field = group_field
        self.reliability = reliability
        self.power = power

    def check(self):
        self.passed = self.p_value < self.reliability

    def execute(self, data):
        raise NotImplementedError


class StatHypothesisTestingWithScipy(StatHypothesisTesting):
    def __init__(
        self,
        target_field: str,
        group_field: str,
        scipy_func,
        reliability: float = 0.05,
        power: float = 0.8,
    ):
        super().__init__(target_field, group_field, reliability, power)
        self.scipy_func = scipy_func

    def execute(self, data):
        grouping_data = list(data.groupby(self.group_field))
        result_stats = self.scipy_func(
            grouping_data[0][1][self.target_field],
            grouping_data[1][1][self.target_field],
        )

        self.statistic = result_stats[0]
        self.p_value = result_stats[1]
        self.check()


class StatTTest(StatHypothesisTestingWithScipy):
    def __init__(
        self,
        target_field: str,
        group_field: str,
        reliability: float = 0.05,
        power: float = 0.8,
    ):
        super().__init__(target_field, group_field, ttest_ind, reliability, power)


class StatKSTest(StatHypothesisTestingWithScipy):
    def __init__(self, target_field: str, group_field: str):
        super().__init__(target_field, group_field, ks_2samp, reliability, power)
