import warnings
from typing import List, Literal, Union

from .experiments import GroupExperiment
from .reporters.matching import MatchingDatasetReporter
from .analyzers.matching import MatchingAnalyzer
from .comparators import TTest, PSI, KSTest
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
        test_mapping = {
            "psi": PSI(grouping_role=TreatmentRole(), compare_by="groups"),
            "ks-test": KSTest(grouping_role=TreatmentRole(), compare_by="groups")
        }
        two_sides = metric == "ate"
        test_pairs = metric == "atc"
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
            TTest(
                compare_by="columns_in_groups",
                baseline_role=AdditionalTargetRole(),
                target_role=TargetRole(),
                grouping_role=TreatmentRole(),
            ),
            MatchingAnalyzer(),
        ]
        if quality_tests != "auto":
            # warnings.warn("Now quality tests aren't supported yet")
            for test in quality_tests:
                executors += [
                    test_mapping[test]
                ]
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
