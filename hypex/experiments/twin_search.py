from hypex.comparators.distances import MahalanobisDistance
from hypex.dataset import TreatmentRole, TargetRole
from hypex.experiments import Experiment
from hypex.ml.faiss import FaissNearestNeighbors
from hypex.operators.operators import MatchingMetrics, SMD
from .preprocessing import PREPROCESSING_DATA

MATCHING = Experiment(
    executors=[
        PREPROCESSING_DATA,
        MahalanobisDistance(grouping_role=TreatmentRole()),
        FaissNearestNeighbors(grouping_role=TreatmentRole(), two_sides=True),
        MatchingMetrics(grouping_role=TreatmentRole(), target_roles=[TargetRole()]),
        SMD(target_roles=[TargetRole()], grouping_role=TreatmentRole()),
    ]
)
