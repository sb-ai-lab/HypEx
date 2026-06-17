from __future__ import annotations

from ..analyzers.aa import OneAAStatAnalyzer
from ..executor.executor import Executor, IfExecutor
from ..utils.enums import ExperimentDataEnum


class IfAAExecutor(IfExecutor):
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
            [
                score_table.loc[:, column].get_values()[0][0]
                for column in score_table.columns
                if "pass" in column
            ]
        )

    def check_rule(self, data, **kwargs) -> bool:
        if self.all_features_passed:
            # A "successful" A/A iteration: no test flagged a difference on any
            # feature ("pass" == pvalue < alpha), so the split is homogeneous.
            return self._count_feature_pass(data) == 0
        if self.sample_size is not None:
            return self._count_feature_pass(data) >= 1
        return False
