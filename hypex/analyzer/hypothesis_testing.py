from abc import ABC

from scipy.stats import ttest_ind, ks_2samp

from hypex.experiment.base import Executor
from hypex.dataset.dataset import ExperimentData, Dataset


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

    def execute(self, data):
        raise NotImplementedError

    def _set_value(self, data: ExperimentData, value: Dataset) -> ExperimentData:
        data.set_value(
            "analysis_tables", 
            self._id,
            self.get_full_name(),
            value
        )
        return data


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

    def execute(self, data: ExperimentData) -> ExperimentData:
        grouping_data = list(data.groupby(self.group_field))
        result_stats = {
            grouping_data[i][0]: self.scipy_func(
                grouping_data[0][1][self.target_field],
                grouping_data[i][1][self.target_field],
            )
            for i in range(1, len(grouping_data))
        }

        # TODO: convert to Dataset in ExperimentData
        result_stats = [
            {
                "group": group,
                "statistic": stats.statistic,
                "p-value": stats.pvalue,
                "pass": stats.pvalue < self.reliability,
            } for group, stats in result_stats.items()
        ]
        
        self._set_value(data, result_stats)
        return data

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
