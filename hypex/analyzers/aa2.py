from typing import Any, Dict

from hypex.dataset import ExperimentData, Dataset, StatisticRole
from hypex.executor import Executor
from hypex.experiments.aa2 import AATest
from hypex.splitters import AASplitter
from hypex.utils import ExperimentDataEnum, ID_SPLIT_SYMBOL
from hypex.reporters.aa import AADictReporter


class AAScoreAnalyzer(Executor):
    AA_SPLITER_CLASS_MAPPING = {c.__name__: c for c in [AASplitter]}

    # TODO: rename alpha
    def __init__(self, alpha: float = 0.05, key: str = ""):
        super().__init__(key=key)
        self.alpha = alpha
        self.__feature_weights = {}

    def _set_value(
            self, data: ExperimentData, value: Any, key: Any = None
    ) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.analysis_tables, self.id, self.key, value
        )

    def _analyze_aa_score(
            self, data: ExperimentData, score_table: Dataset
    ) -> ExperimentData:
        search_flag = f"{ID_SPLIT_SYMBOL}p-value{ID_SPLIT_SYMBOL}"
        self.__feature_weights = {
            c: 1 - abs(self.alpha - score_table.loc[:, c].mean())
            for c in score_table.columns
            if search_flag in c
        }
        aa_scores = {
            c.replace(f"{ID_SPLIT_SYMBOL}p-value", ""): v
            for c, v in self.__feature_weights.items()
        }
        aa_passed = {c: v >= (1 - self.alpha * 0.2) for c, v in aa_scores.items()}
        result = Dataset.from_dict({"score": aa_scores, "pass": aa_passed}, roles={})
        self.key = "aa score"
        return self._set_value(data, result)

    def build_splitter_from_id(self, splitter_id: str):
        return self.AA_SPLITER_CLASS_MAPPING.get(splitter_id[:splitter_id.find(ID_SPLIT_SYMBOL)]).build_from_id(
            splitter_id)

    def _geet_best_split(self, data: ExperimentData, score_table: Dataset) -> Dict[str, Any]:
        aa_split_scores = score_table.apply(
            lambda x: (
                    sum([x[k] * v for k, v in self.__feature_weights.items()])
                    / len(self.__feature_weights)
                    * 2
                    / 3
                    + x["mean test score"] / 3
            ),
            axis=1,
            role={"aa split score": StatisticRole()},
        )
        best_index = aa_split_scores.idxmax()
        score_dict = score_table.loc[best_index, :].transpose().to_records()[0]
        best_score_stat = AADictReporter.convert_flat_dataset(score_dict)
        self.key = "best split statistics"
        result = self._set_value(data, best_score_stat)
        return {
            "index": best_index,
            "data": result
        }

    def _set_best_split(self, data: ExperimentData, score_table: Dataset, best_index: int) -> ExperimentData:
        self.key = "best splitter"
        # TODO: replace get_values
        best_splitter_id = score_table.loc[best_index, "splitter_id"].to_dict()["data"][
            "data"
        ][0][0]
        result = data.set_value(
            ExperimentDataEnum.variables, self.id, self.key, best_splitter_id, self.key
        )
        best_splitter = self.build_splitter_from_id(best_splitter_id)
        best_splitter.save_groups = False
        result = best_splitter.execute(result)
        return result

    def _analyze_best_split(
            self, data: ExperimentData, score_table: Dataset
    ) -> ExperimentData:
        best_split = self._geet_best_split(data, score_table)
        return self._set_best_split(best_split["data"], score_table, best_split["index"])

    def execute(self, data: ExperimentData) -> ExperimentData:
        score_table_id = data.get_one_id(AATest, ExperimentDataEnum.analysis_tables)
        score_table = data.analysis_tables[score_table_id]

        data = self._analyze_aa_score(data, score_table)
        data = self._analyze_best_split(data, score_table)
        return data
