from hypex.comparators.distances import MahalanobisDistance
from hypex.dataset import TreatmentRole, TargetRole
from hypex.encoders.encoders import DummyEncoder
from hypex.experiments import Experiment
from hypex.ml.faiss import FaissNearestNeighbors
from hypex.operators.operators import MatchingMetrics


class Matching(ExperimentShell): 

    @staticmethod
    def _make_experiment(filters, distance, two_sides, metric, quality_tests):
        filters_mapping = {"dummy-encoder": DummyEncoder(), "auto": DummyEncoder()}
        distance_mapping = {"mahalanobis": MahalanobisDistance(grouping_role=TreatmentRole())}
        tests_mapping = {}
        filters = (
            filters
            if isinstance(filters, List)
            else [filters]
        )
        if filters not in filters_mapping: 
            warn.warning("Сurrently only dummy encoder is supported")
        executors = []
        for i in filters:
            executors += [filters_mapping[i]]
        executors += [distance_mapping[distance], 
                      FaissNearestNeighbors(grouping_role=TreatmentRole(), two_sides=two_sides),
                      MatchingMetrics(grouping_role=TreatmentRole(), target_roles=[TargetRole()]), metric=metric]
        if quality_tests != "auto": 
            warn.warning("Now quality tests aren't supported yet")
        return Experiment(
            executors=executors
        )
    
    def __init__(
        self,
        filters: Union[Literal["fillna", "const-filter", "na-filter", "dummy-encoder", "auto"], 
                             List[Literal["fillna", "const-filter", "na-filter", "dummy-encoder", "auto"]]
                            ] = "auto",
        distance: Literal["mahalanobis", "l2"] = "mahalanobis",
        two_sides: bool = True,
        metric: Literal["atc", "att", "ate", "auto"] = "auto",
        quality_tests: Union[Literal["smd", "psi", "ks-test", "repeats", "auto"], 
                             List[Literal["smd", "psi", "ks-test", "repeats", "auto"]]
                            ] = "auto"
    ):
        super().__init__(
                experiment=self._make_experiment(filters, distance, two_sides, metric, quality_tests),
                output=MatchingOutput(),
            )
        


