from typing import Dict, List

# from hypex.splitters import AASplitter
from hypex.comparators import KSTest, TTest, Chi2Test
from hypex.dataset import Dataset, ExperimentData, StatisticRole
from hypex.executor import Executor
from hypex.experiments.base_complex import ParamsExperiment
from hypex.utils import BackendsEnum, ExperimentDataEnum


class OneAAStatAnalyzer(Executor):
    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.analysis_tables, self.id, self.__class__.__name__, value
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        analysis_tests: List[type] = [TTest, KSTest, Chi2Test]
        executor_ids = data.get_ids(analysis_tests)

        analysis_data: Dict[str, float] = {}
        for c, spaces in executor_ids.items():
            analysis_ids = spaces.get("analysis_tables", [])
            if len(analysis_ids) > 0:
                if len(analysis_ids) > 1:
                    t_data = data.analysis_tables[analysis_ids[0]].append(
                        [data.analysis_tables[k] for k in analysis_ids[1:]]
                    )
                else:
                    t_data = data.analysis_tables[analysis_ids[0]]
                # t_data.data.index = analysis_ids
                for f in ["p-value", "pass"]:
                    analysis_data[f"mean {c.__name__} {f}"] = t_data[f].mean()
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
            {f: StatisticRole() for f in analysis_data},
            BackendsEnum.pandas,
        )

        return self._set_value(data, analysis_dataset)


class AAScoreAnalyzer(Executor):
    AA_SPLITER_CLASS_MAPPING = {c.__name__: c for c in [AASplitter]}

    # TODO: rename alpha
    def __init__(self, alpha: float = 0.05, key: str = ""):
        super().__init__(key=key)
        self.alpha = alpha

    def execute(self, data: ExperimentData) -> ExperimentData:
        score_table = data.get_one_id(
            ParamsExperiment, ExperimentDataEnum.analysis_tables
        )
        best_score_stat = data.ds.loc[data.ds["mean test score"].idxmax(), :]


class AAStatAnalyzer(Executor):
    pass
