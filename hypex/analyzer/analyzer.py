from abc import ABC
from typing import Iterable, Dict

from hypex.comparators.comparators import GroupDifference, GroupSizes
from hypex.comparators.hypothesis_testing import TTest, KSTest
from hypex.dataset.dataset import ExperimentData
from hypex.experiment.experiment import (
    Experiment,
    ComplexExecutor,
    OnTargetExperiment,
    Executor,
)
from hypex.splitters.aa_splitter import AASplitter
from hypex.stats.descriptive import Mean


class OneAASplit(ComplexExecutor):
    default_inner_executors = {"mean": Mean()}

    # TODO: implement
    def execute(self, data: ExperimentData) -> ExperimentData:
        # TODO: class or text?
        executor_ids = data.get_ids([TTest, KSTest])

        # meta_data = {
        #     TTest: {"p-value": [], "passed": []},
        #     KSTest: {"p-value": [], "passed": []},
        # }
        # for key, value in executor_ids.items():
        #     for v in value:
        #         meta_data[key]["p-value"] += list(data.analysis_tables[v]["p-value"])
        #         meta_data[key]["passed"] += list(data.analysis_tables[v]["pass"])

        # result = {}
        # for key, value in meta_data.items():
        #     result[f"{key.__name__} p-value"] = self.inner_executors["mean"].calc(
        #         value["p-value"]
        #     )
        #     result[f"{key.__name__} passed %"] = (
        #         self.inner_executors["mean"].calc(value["passed"]) * 100
        #     )
        
        # return data
