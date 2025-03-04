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
        key: str = "",
    ):
        self.sample_size = sample_size
        super().__init__(if_executor, else_executor, key)

    def check_rule(self, data, **kwargs) -> bool:
        if self.sample_size is not None:
            score_table_id = data.get_one_id(
                OneAAStatAnalyzer,
                ExperimentDataEnum.analysis_tables,
            )
            score_table = data.analysis_tables[score_table_id]
            feature_pass = sum(
                [
                    score_table.loc[:, column].get_values()[0][0]
                    for column in score_table.columns
                    if "pass" in column
                ]
            )
            return True if feature_pass >= 1 else False
        return False
