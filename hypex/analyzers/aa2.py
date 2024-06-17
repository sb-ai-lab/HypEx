from typing import Any

from hypex.dataset import ExperimentData
from hypex.executor import Executor
from hypex.experiments.aa2 import AATest
from hypex.splitters import AASplitter
from hypex.utils import ExperimentDataEnum, SetParamsDictTypes
from hypex.reporters.aa import AADictReporter


class AAScoreAnalyzer(Executor):
    AA_SPLITER_CLASS_MAPPING = {c.__name__: c for c in [AASplitter]}

    # TODO: rename alpha
    def __init__(self, alpha: float = 0.05, key: str = ""):
        super().__init__(key=key)
        self.alpha = alpha

    def _set_value(
        self, data: ExperimentData, value: Any, key: Any = None
    ) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.analysis_tables, self.id, self.key, value
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        score_table_id = data.get_one_id(AATest, ExperimentDataEnum.analysis_tables)
        score_table = data.analysis_tables[score_table_id]
        score_dict = score_table.loc[
            score_table["mean test score"].idxmax(), :
        ].transpose().to_records()[0]
        best_score_stat = AADictReporter.convert_flat_dataset(score_dict)
        self.key = "best score statistics"
        self._set_value(data, best_score_stat)
        return data
