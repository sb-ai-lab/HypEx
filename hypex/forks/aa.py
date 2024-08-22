from typing import Optional

from hypex.dataset import ExperimentData
from hypex.executor.executor import IfExecutor
from hypex.experiments.base_complex import ParamsExperiment
from hypex.utils.constants import ID_SPLIT_SYMBOL
from hypex.utils.enums import ExperimentDataEnum


class IfAAExecutor(IfExecutor):
    def __init__(
        self,
        if_executor,
        else_executor = None,
        sample_size: Optional[float] = None,
        key: str = "",
    ):
        self.sample_size = sample_size
        super().__init__(if_executor, else_executor, key)

    def execute(self, data: ExperimentData) -> ExperimentData:
        search_flag = f"{ID_SPLIT_SYMBOL}pass{ID_SPLIT_SYMBOL}"
        score_table_id = data.get_one_id(
            ParamsExperiment, ExperimentDataEnum.analysis_tables, "AATest"
        )
        score_table = data.analysis_tables[score_table_id]
        feature_pass = sum([
            True if score_table.loc[:, column].sum() else False
            for column in score_table.columns
            if search_flag in column
        ])
        if self.sample_size is not None and feature_pass > 0:
            return self.if_executor.execute(data)
        return data
