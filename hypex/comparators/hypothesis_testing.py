from abc import ABC
from typing import Dict, Union, Any, List

from scipy.stats import ttest_ind

from hypex.experiment.experiment import Executor
from hypex.comparators.comparators import GroupComparator
from hypex.dataset.dataset import ExperimentData, Dataset
from hypex.dataset.roles import RolesType, FromDictType, StatisticRole



class StatHypothesisTestingWithScipy(GroupComparator):
    def __init__(
        self,
        reliability: float = 0.05,
        inner_executors: Union[Dict[str, Executor], None] = None,
        full_name: Union[str, None] = None,
        key: Any = 0,
    ):
        super().__init__(inner_executors, full_name, key)
        self.reliability = reliability

    # excessive override
    def _extract_dataset(self, compare_result: FromDictType, roles: Union[RolesType, None]=None) -> Dataset:
        result_stats: List[Dict[str, Any]] = [
            {
                "group": group,
                "statistic": stats.statistic,
                "p-value": stats.pvalue,
                "pass": stats.pvalue < self.reliability,
            }
            for group, stats in compare_result.items()
        ]
        # mypy does not see an heir 
        # return super()._extract_dataset(
        #     result_stats,
        #     roles={StatisticRole(): ["group", "statistic", "p-value", "pass"]}
        # )

        return super()._extract_dataset(
            result_stats,
            roles={f: StatisticRole() for f in ["group", "statistic", "p-value", "pass"]}
        )


class TTest(StatHypothesisTestingWithScipy):
    def _comparison_function(self, control_data, test_data) -> ExperimentData:
        return ttest_ind(control_data, test_data)


class KSTest(StatHypothesisTestingWithScipy):
    def _comparison_function(self, control_data, test_data) -> ExperimentData:
        return ttest_ind(control_data, test_data)
