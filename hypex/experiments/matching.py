import warnings
from typing import List, Literal, Union

from hypex.analyzers.matching import MatchingAnalyzer
from hypex.comparators.distances import MahalanobisDistance
from hypex.dataset import TreatmentRole, TargetRole
from hypex.encoders.encoders import DummyEncoder
from hypex.experiments import Experiment
from hypex.ml.faiss import FaissNearestNeighbors
from hypex.operators.operators import MatchingMetrics
from hypex.ui.base import ExperimentShell
from hypex.ui.matching import MatchingOutput


class Matching(ExperimentShell):

    @staticmethod
    def _make_experiment(filters, distance, two_sides, metric, quality_tests):
        filters_mapping = {"dummy-encoder": DummyEncoder(), "auto": DummyEncoder()}
        distance_mapping = {
            "mahalanobis": MahalanobisDistance(grouping_role=TreatmentRole())
        }
        filters = filters if isinstance(filters, List) else [filters]
        if any(filter_ not in filters_mapping for filter_ in filters):
            warnings.warn("Сurrently only dummy encoder is supported")
        executors = []
        for i in filters:
            executors += [filters_mapping[i]]
        executors += [
            FaissNearestNeighbors(grouping_role=TreatmentRole(), two_sides=two_sides),
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
        filters: Union[
            Literal["fillna", "const-filter", "na-filter", "dummy-encoder", "auto"],
            List[
                Literal["fillna", "const-filter", "na-filter", "dummy-encoder", "auto"]
            ],
            None,
        ] = "auto",
        distance: Literal["mahalanobis", "l2"] = "mahalanobis",
        two_sides: bool = True,
        metric: Literal["atc", "att", "ate", "auto"] = "auto",
        quality_tests: Union[
            Literal["smd", "psi", "ks-test", "repeats", "auto"],
            List[Literal["smd", "psi", "ks-test", "repeats", "auto"]],
        ] = "auto",
    ):
        super().__init__(
            experiment=self._make_experiment(
                filters, distance, two_sides, metric, quality_tests
            ),
            output=MatchingOutput(),
        )
