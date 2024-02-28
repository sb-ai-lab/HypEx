from abc import ABC

from scipy.stats import ttest_ind, ks_2samp

from hypex.experiment.base import Executor
from hypex.dataset.dataset import ExperimentData, Dataset
from hypex.comparators.comparators import Comparator

class StatHypothesisTestingWithScipy(ABC, Comparator):

    def __init__(
        self,
        target_field: str,
        comparison_function: Callable,
        reliability: float = 0.05,
    ):
        super().__init__(target_field, comparison_function)
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
        # TODO: not implemented
        return Dataset(result_stats)

class StatTTest(StatHypothesisTestingWithScipy):
    def _comparison_function(self, control_data, test_data) -> ExperimentData:
        return ttest_ind(control_data, test_data)

class StatKSTest(StatHypothesisTestingWithScipy):
    def _comparison_function(self, control_data, test_data) -> ExperimentData:
        return ttest_ind(control_data, test_data)
