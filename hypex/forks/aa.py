from typing import Optional

from hypex.analyzers.aa import OneAAStatAnalyzer
from hypex.dataset import ExperimentData
from hypex.executor.executor import IfExecutor
from hypex.utils.enums import ExperimentDataEnum


class IfAAExecutor(IfExecutor):
    def __init__(
        self,
        if_executor=None,
        else_executor=None,
        sample_size: Optional[float] = None,
        key: str = "",
    ):
        self.sample_size = sample_size
        super().__init__(if_executor, else_executor, key)

    def execute(self, data: ExperimentData) -> ExperimentData:
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
            if feature_pass >= 1:
                return (
                    self.if_executor.execute(data)
                    if self.if_executor is not None
                    else True
                )
        return data if self.if_executor is not None else False
