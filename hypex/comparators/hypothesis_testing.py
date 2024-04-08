from abc import ABC
from typing import Dict, Union, Any, List

# mypy import-untyped
from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu

from hypex.comparators.comparators import GroupComparator
from hypex.dataset.dataset import ExperimentData, Dataset
from hypex.dataset.roles import ABCRole, StatisticRole
from hypex.experiments.base import Executor
from hypex.utils.enums import SpaceEnum


class StatHypothesisTestingWithScipy(GroupComparator, ABC):
    def __init__(
        self,
        grouping_role: Union[ABCRole, None] = None,
        space: SpaceEnum = SpaceEnum.auto,
        reliability: float = 0.05,
        inner_executors: Union[Dict[str, Executor], None] = None,
        full_name: Union[str, None] = None,
        key: Any = "",
    ):
        super().__init__(grouping_role, space, inner_executors, full_name, key)
        self.reliability = reliability

    # excessive override
    def _local_extract_dataset(
        self, compare_result: Dict[Any, Any], roles=None
    ) -> Dataset:
        # stats type
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
            roles={
                f: StatisticRole() for f in ["group", "statistic", "p-value", "pass"]
            },
        )


class TTest(StatHypothesisTestingWithScipy):
    def _comparison_function(self, control_data, test_data) -> ExperimentData:
        return ttest_ind(
            control_data.data.values.flatten(), test_data.data.values.flatten()
        )


class KSTest(StatHypothesisTestingWithScipy):
    def _comparison_function(self, control_data, test_data) -> ExperimentData:
        return ks_2samp(
            control_data.data.values.flatten(), test_data.data.values.flatten()
        )


class MannWhitney(StatHypothesisTestingWithScipy):
    def _comparison_function(self, control_data, test_data):
        return mannwhitneyu(
            control_data.data.values.flatten(), test_data.data.values.flatten()
        )
