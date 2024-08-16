import warnings
from typing import List, Literal, Union

from hypex.analyzers.matching import MatchingAnalyzer
from hypex.comparators.distances import MahalanobisDistance
from hypex.dataset import TreatmentRole, TargetRole
from hypex.executor import Executor
from hypex.experiments.base import Experiment
from hypex.ml.faiss import FaissNearestNeighbors
from hypex.operators.operators import MatchingMetrics, Bias
from hypex.ui.base import ExperimentShell
from hypex.ui.matching import MatchingOutput


class Matching(ExperimentShell):

    @staticmethod
    def _make_experiment(
        distance: Literal["mahalanobis", "l2"] = "mahalanobis",
        two_sides: bool = True,
        metric: Literal["atc", "att", "ate", "auto"] = "auto",
        bias_estimation: bool = True,
        quality_tests: Union[
            Literal["smd", "psi", "ks-test", "repeats", "auto"],
            List[Literal["smd", "psi", "ks-test", "repeats", "auto"]],
        ] = "auto",
    ) -> Experiment:
        distance_mapping = {
            "mahalanobis": MahalanobisDistance(grouping_role=TreatmentRole())
        }
        executors: List[Executor] = [
            FaissNearestNeighbors(grouping_role=TreatmentRole(), two_sides=two_sides)
        ]
        if bias_estimation:
            executors += [
                Bias(grouping_role=TreatmentRole(), target_roles=[TargetRole()]),
            ]
        if metric in ["atc", "ate"] and not two_sides:
            raise ValueError(f"Can not estimate {metric} while two_sides is False")
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
        return Experiment(
            executors=(
                executors
                if distance == "l2"
                else [distance_mapping[distance]] + executors
            )
        )

    def __init__(
        self,
        distance: Literal["mahalanobis", "l2"] = "mahalanobis",
        two_sides: bool = True,
        metric: Literal["atc", "att", "ate", "auto"] = "auto",
        bias_estimation: bool = True,
        quality_tests: Union[
            Literal["smd", "psi", "ks-test", "repeats", "auto"],
            List[Literal["smd", "psi", "ks-test", "repeats", "auto"]],
        ] = "auto",
    ):
        super().__init__(
            experiment=self._make_experiment(
                distance, two_sides, metric, bias_estimation, quality_tests
            ),
            output=MatchingOutput(),
        )
