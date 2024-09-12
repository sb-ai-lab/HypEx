import warnings
from typing import List, Literal, Union

from hypex.analyzers.matching import MatchingAnalyzer
from hypex.comparators.distances import MahalanobisDistance
from hypex.dataset import TreatmentRole, TargetRole
from hypex.executor import Executor
from hypex.experiments import GroupExperiment
from hypex.experiments.base import Experiment
from hypex.ml.faiss import FaissNearestNeighbors
from hypex.operators.operators import MatchingMetrics, Bias
from hypex.reporters.matching import MatchingDatasetReporter
from hypex.ui.base import ExperimentShell
from hypex.ui.matching import MatchingOutput


class Matching(ExperimentShell):

    @staticmethod
    def _make_experiment(
        group_match: bool = False,
        distance: Literal["mahalanobis", "l2"] = "mahalanobis",
        metric: Literal["atc", "att", "ate"] = "ate",
        bias_estimation: bool = True,
        quality_tests: Union[
            Literal["smd", "psi", "ks-test", "repeats", "auto"],
            List[Literal["smd", "psi", "ks-test", "repeats", "auto"]],
        ] = "auto",
    ) -> Experiment:
        distance_mapping = {
            "mahalanobis": MahalanobisDistance(grouping_role=TreatmentRole())
        }
        two_sides = True if metric == "ate" else False
        test_pairs = True if metric == "atc" else False
        executors: List[Executor] = [
            FaissNearestNeighbors(
                grouping_role=TreatmentRole(),
                two_sides=two_sides,
                test_pairs=test_pairs,
            )
        ]
        if bias_estimation:
            executors += [
                Bias(grouping_role=TreatmentRole(), target_roles=[TargetRole()]),
            ]
        executors += [
            MatchingMetrics(
                grouping_role=TreatmentRole(),
                target_roles=[TargetRole()],
                metric=metric,
            ),
            MatchingAnalyzer(),
        ]
        if quality_tests != "auto":
            warnings.warn("Now quality tests aren't supported yet")
        return (
            Experiment(
                executors=(
                    executors
                    if distance == "l2"
                    else [distance_mapping[distance]] + executors
                )
            )
            if not group_match
            else GroupExperiment(
                executors=(
                    executors
                    if distance == "l2"
                    else [distance_mapping[distance]] + executors
                ),
                reporter=MatchingDatasetReporter(),
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
            experiment=self._make_experiment(
                group_match, distance, metric, bias_estimation, quality_tests
            ),
            output=MatchingOutput(GroupExperiment if group_match else MatchingAnalyzer),
        )
