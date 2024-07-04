from hypex.comparators.distances import MahalanobisDistance
from hypex.dataset import TreatmentRole, TargetRole
from hypex.experiments import Experiment
from hypex.ml.faiss import FaissNearestNeighbors
from hypex.operators.operators import MatchingMetrics

TWIN_SEARCH = Experiment(
    executors=[
        MahalanobisDistance(grouping_role=TreatmentRole()),
        FaissNearestNeighbors(grouping_role=TreatmentRole(), two_sides=True),
        MatchingMetrics(grouping_role=TreatmentRole(), target_roles=[TargetRole()]),
    ]
)
