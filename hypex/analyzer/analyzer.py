from abc import ABC

from hypex.experiment.experiment import Experiment, ComplexExecutor
from hypex.stats.descriptive import Mean


class Analyzer(ABC, Experiment, ComplexExecutor):
    default_executors = []

    def __init__(
        self,
        executors: Iterable[Executor] = None,
        inner_executors: Dict[str, Executor] = None,
        full_name: str = None,
        index: int = 0,
    ):
        self.inner_executors: Dict[str, Executor] = (
            self.get_inner_executors(inner_executors),
        )
        self.executors = executors or self.default_executors

        super().__init__(
            executors=executors, transformer=False, full_name=full_name, index=index
        )


class OneAASplit(Analyzer):
    default_executors = [
        AASplitter(),
        OnTargetExperiment([TTest(), KSTest(), GroupDifference()]),
        GroupSizes(),
    ]
    default_inner_executors = {"mean": Mean()}

    def execute(self, data: ExperimentData) -> ExperimentData:
        executor_ids = self.get_executor_ids([TTest, KSTest])
        meta_data = {
            TTest: {"p-value": [], "passed": []},
            KSTest: {"p-value": [], "passed": []},
        }
        for key, value in executor_ids.items():
            for v in value:
                meta_data[key]["p-value"] += list(data.analysis_tables[v]["p-value"])
                meta_data[key]["passed"] += list(data.analysis_tables[v]["pass"])

        result = {}
        for key, value in meta_data.items():
            result[f"{key.__name__} p-value"] = self.inner_executors["mean"].calc(
                value["p-value"]
            )
            result[f"{key.__name__} passed %"] = self.inner_executors["mean"].calc(
                value["passed"]
            )*100
