from hypex.dataset import ExperimentData
from hypex.executor import Executor
from hypex.experiments.aa2 import AATest
from hypex.splitters import AASplitter
from hypex.utils import ExperimentDataEnum


class AAScoreAnalyzer(Executor):
    AA_SPLITER_CLASS_MAPPING = {c.__name__: c for c in [AASplitter]}

    # TODO: rename alpha
    def __init__(self, alpha: float = 0.05, key: str = ""):
        super().__init__(key=key)
        self.alpha = alpha

    def execute(self, data: ExperimentData) -> ExperimentData:
        score_table_id = data.get_one_id(
            AATest, ExperimentDataEnum.analysis_tables
        )
        score_table = data.analysis_tables[score_table_id]
        best_score_stat = score_table.loc[score_table["mean test score"].idxmax(), :]
