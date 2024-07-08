from hypex.comparators.distances import MahalanobisDistance
from hypex.dataset import TreatmentRole, TargetRole
from hypex.encoders.encoders import DummyEncoder
from hypex.experiments import Experiment
from hypex.ml.faiss import FaissNearestNeighbors
from hypex.operators.operators import MatchingMetrics

MATCHING = Experiment(
    executors=[
        DummyEncoder(),
        MahalanobisDistance(grouping_role=TreatmentRole()),
        FaissNearestNeighbors(grouping_role=TreatmentRole(), two_sides=True),
        MatchingMetrics(grouping_role=TreatmentRole(), target_roles=[TargetRole()]),
    ]
)
