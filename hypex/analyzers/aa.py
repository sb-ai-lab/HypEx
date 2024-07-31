from typing import Dict, List, Any

from hypex.comparators import KSTest, TTest, Chi2Test
from hypex.dataset import Dataset, ExperimentData, StatisticRole
from hypex.executor import Executor
from hypex.experiments.base_complex import ParamsExperiment
from hypex.reporters.aa import OneAADictReporter
from hypex.splitters import AASplitter, AASplitterWithStratification
from hypex.utils import BackendsEnum, ExperimentDataEnum, ID_SPLIT_SYMBOL


class OneAAStatAnalyzer(Executor):
    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        return data.set_value(ExperimentDataEnum.analysis_tables, self.id, value)

    def execute(self, data: ExperimentData) -> ExperimentData:
        analysis_tests: List[type] = [TTest, KSTest, Chi2Test]
        executor_ids = data.get_ids(
            analysis_tests, searched_space=ExperimentDataEnum.analysis_tables
        )

        analysis_data: Dict[str, float] = {}
        for class_, spaces in executor_ids.items():
            analysis_ids = spaces.get("analysis_tables", [])
            if len(analysis_ids) > 0:
                if len(analysis_ids) > 1:
                    t_data = data.analysis_tables[analysis_ids[0]].append(
                        [data.analysis_tables[k] for k in analysis_ids[1:]]
                    )
                else:
                    t_data = data.analysis_tables[analysis_ids[0]]
                # t_data.data.index = analysis_ids
                for field in ["p-value", "pass"]:
                    analysis_data[f"mean {class_} {field}"] = t_data[field].mean()
        analysis_data["mean test score"] = (
            analysis_data["mean TTest p-value"]
            + 2 * analysis_data["mean KSTest p-value"]
        )
        if "mean Chi2Test p-value" in analysis_data:
            analysis_data["mean test score"] += (
                2 * analysis_data["mean Chi2Test p-value"]
            )
            analysis_data["mean test score"] /= 5
        else:
            analysis_data["mean test score"] /= 3

        analysis_dataset = Dataset.from_dict(
            [analysis_data],
            {field: StatisticRole() for field in analysis_data},
            BackendsEnum.pandas,
        )

        return self._set_value(data, analysis_dataset)


class AAScoreAnalyzer(Executor):
    AA_SPLITER_CLASS_MAPPING = {
        class_.__name__: class_ for class_ in [AASplitter, AASplitterWithStratification]
    }

    # TODO: rename alpha
    def __init__(self, alpha: float = 0.05, key: str = ""):
        super().__init__(key=key)
        self.alpha = alpha
        self.__feature_weights = {}

    def _set_value(
        self, data: ExperimentData, value: Any, key: Any = None
    ) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.analysis_tables,
            executor_id=self.id,
            key=self.key,
            value=value,
        )

    def _analyze_aa_score(
        self, data: ExperimentData, score_table: Dataset
    ) -> ExperimentData:
        search_flag = f"{ID_SPLIT_SYMBOL}p-value{ID_SPLIT_SYMBOL}"
        self.__feature_weights = {
            column: 1 - abs(self.alpha - score_table.loc[:, column].mean())
            for column in score_table.columns
            if search_flag in column
        }
        aa_scores = {
            class_.replace(f"{ID_SPLIT_SYMBOL}p-value", ""): value
            for class_, value in self.__feature_weights.items()
        }
        aa_passed = {
            class_: value >= (1 - self.alpha * 1.2)
            for class_, value in aa_scores.items()
        }
        result = Dataset.from_dict({"score": aa_scores, "pass": aa_passed}, roles={})
        self.key = "aa score"
        return self._set_value(data, result)

    def build_splitter_from_id(self, splitter_id: str):
        splitter_class = self.AA_SPLITER_CLASS_MAPPING.get(
            splitter_id[: splitter_id.find(ID_SPLIT_SYMBOL)]
        )
        if splitter_class is None:
            raise ValueError(f"{splitter_id} is not a valid splitter id")
        return splitter_class.build_from_id(splitter_id)

    def _get_best_split(
        self, data: ExperimentData, score_table: Dataset
    ) -> Dict[str, Any]:
        aa_split_scores = score_table.apply(
            lambda x: (
                (
                    (
                        (
                            sum(
                                x[key] * value
                                for key, value in self.__feature_weights.items()
                            )
                            / len(self.__feature_weights)
                        )
                        * 2
                    )
                    / 3
                )
                + x["mean test score"] / 3
            ),
            axis=1,
            role={"aa split score": StatisticRole()},
        )
        best_index = aa_split_scores.idxmax()
        score_dict = score_table.loc[best_index, :].transpose().to_records()[0]
        best_score_stat = OneAADictReporter.convert_flat_dataset(score_dict)
        self.key = "best split statistics"
        result = self._set_value(data, best_score_stat)
        return {"index": best_index, "data": result}

    def _set_best_split(
        self, data: ExperimentData, score_table: Dataset, best_index: int
    ) -> ExperimentData:
        self.key = "best splitter"
        best_splitter_id = score_table.loc[best_index, "splitter_id"].get_values(0, 0)
        result = data.set_value(
            ExperimentDataEnum.variables, self.id, best_splitter_id, self.key
        )
        best_splitter = self.build_splitter_from_id(best_splitter_id)
        best_splitter.save_groups = False
        best_splitter.constant_key = False
        best_splitter.key = "best"
        result = best_splitter.execute(result)
        return result

    def _analyze_best_split(
        self, data: ExperimentData, score_table: Dataset
    ) -> ExperimentData:
        best_split = self._get_best_split(data, score_table)
        return self._set_best_split(
            best_split["data"], score_table, best_split["index"]
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        score_table_id = data.get_one_id(
            ParamsExperiment, ExperimentDataEnum.analysis_tables, "AATest"
        )
        score_table = data.analysis_tables[score_table_id]

        data = self._analyze_aa_score(data, score_table)
        data = self._analyze_best_split(data, score_table)
        return data
