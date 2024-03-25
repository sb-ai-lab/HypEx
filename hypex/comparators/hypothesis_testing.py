from abc import ABC

from scipy.stats import ttest_ind

from hypex.comparators.comparators import GroupComparator
from hypex.dataset.dataset import ExperimentData, Dataset


class StatHypothesisTestingWithScipy(ABC, GroupComparator):
    def __init__(
        self,
        reliability: float = 0.05,
        full_name: str = None,
        index: int = 0,
    ):
        super().__init__(full_name, index)
        self.reliability = reliability

    def _extract_dataset(self, compare_result: Dict) -> Dataset:
        result_stats = [
            {
                "group": group,
                "statistic": stats.statistic,
                "p-value": stats.pvalue,
                "pass": stats.pvalue < self.reliability,
            } for group, stats in result_stats.items()
        ]
        return super()._extract_dataset(result_stats)

class TTest(StatHypothesisTestingWithScipy):
    def _comparison_function(self, control_data, test_data) -> ExperimentData:
        return ttest_ind(control_data, test_data)

class KSTest(StatHypothesisTestingWithScipy):
    def _comparison_function(self, control_data, test_data) -> ExperimentData:
        return ttest_ind(control_data, test_data)
