import warnings
from typing import List, Literal, Union

from .experiments import GroupExperiment
from .reporters.matching import MatchingDatasetReporter
from .analyzers.matching import MatchingAnalyzer
from .comparators import TTest
from .comparators.distances import MahalanobisDistance
from .dataset import TreatmentRole, TargetRole, AdditionalTargetRole
from .executor import Executor
from .experiments.base import Experiment
from .ml.faiss import FaissNearestNeighbors
from .operators.operators import MatchingMetrics, Bias
from .ui.base import ExperimentShell
from .ui.matching import MatchingOutput


class Matching(ExperimentShell):

    @staticmethod
    def create_experiment(self, **kwargs) -> Experiment:
        distance_mapping = {
            "mahalanobis": MahalanobisDistance(grouping_role=TreatmentRole())
        }
        two_sides = kwargs.get("metric") == "ate"
        test_pairs = kwargs.get("metric") == "atc"
        executors: List[Executor] = [
            FaissNearestNeighbors(
                grouping_role=TreatmentRole(),
                two_sides=two_sides,
                test_pairs=test_pairs,
            )
        ]
        if kwargs.get("bias_estimation"):
            executors += [
                Bias(grouping_role=TreatmentRole(), target_roles=[TargetRole()]),
            ]
        executors += [
            MatchingMetrics(
                grouping_role=TreatmentRole(),
                target_roles=[TargetRole()],
                metric=kwargs.get("metric"),
            ),
            TTest(
                compare_by="columns_in_groups",
                baseline_role=AdditionalTargetRole(),
                target_role=TargetRole(),
                grouping_role=TreatmentRole(),
            ),
            MatchingAnalyzer(),
        ]
        if kwargs.get("quality_tests") != "auto":
            warnings.warn("Now quality tests aren't supported yet")
        return (
            GroupExperiment(
                executors=(
                    executors
                    if kwargs.get("distance") == "l2"
                    else [distance_mapping[kwargs.get("distance")]] + executors
                ),
                reporter=MatchingDatasetReporter(),
            )
            if kwargs.get("group_match")
            else Experiment(
                executors=(
                    executors
                    if kwargs.get("distance") == "l2"
                    else [distance_mapping[kwargs.get("distance")]] + executors
                )
            )
        )

    def __init__(
        self,
        group_match: bool = False,
        distance: Literal["mahalanobis", "l2"] = "mahalanobis",
        metric: Literal["atc", "att", "ate"] = "ate",
        bias_estimation: bool = True,
        quality_tests: Union[
            Literal["smd", "psi", "ks-test", "repeats", "auto"],
            List[Literal["smd", "psi", "ks-test", "repeats", "auto"]],
        ] = "auto",
    ):
        super().__init__(
            output=MatchingOutput(GroupExperiment if group_match else MatchingAnalyzer),
            create_experiment_kwargs={
                "group_match": group_match,
                "distance": distance,
                "metric": metric,
                "bias_estimation": bias_estimation,
                "quality_tests": quality_tests,
            },
        )
