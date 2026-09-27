from __future__ import annotations

from ..analyzers.aa import OneAAStatAnalyzer
from ..executor.executor import Executor, IfExecutor
from ..utils.enums import ExperimentDataEnum


class IfAAExecutor(IfExecutor):
    """Conditional executor for A/A test stopping criteria.

    Args:
        if_executor: Executor to run when the rule is met.
        else_executor: Executor to run when the rule is not met.
        sample_size: Fraction of data sampled per iteration (legacy rule).
        all_features_passed: If True, stop when no test flags a
            difference on any feature (early-stopping rule).
        key: Optional identifier key.
    """

    def __init__(
        self,
        if_executor: Executor | None = None,
        else_executor: Executor | None = None,
        sample_size: float | None = None,
        all_features_passed: bool = False,
        key: str = "",
    ):
        self.sample_size = sample_size
        self.all_features_passed = all_features_passed
        super().__init__(if_executor, else_executor, key)

    def _count_feature_pass(self, data) -> float:
        score_table_id = data.get_one_id(
            OneAAStatAnalyzer,
            ExperimentDataEnum.analysis_tables,
        )
        score_table = data.analysis_tables[score_table_id]
        return sum(
            score_table.select(column).iget_values(0, 0)
            for column in score_table.columns
            if "pass" in column
        )

    def check_rule(self, data, **kwargs) -> bool:
        if self.all_features_passed:
            # "pass" == p < alpha → difference detected.
            # A clean split has ZERO passes across all features.
            return self._count_feature_pass(data) == 0
        if self.sample_size is not None:
            return self._count_feature_pass(data) >= 1
        return False
