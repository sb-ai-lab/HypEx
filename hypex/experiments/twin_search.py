from hypex.comparators.distances import MahalanobisDistance
from hypex.dataset import TreatmentRole
from hypex.experiments import Experiment
from hypex.ml.faiss import FaissNearestNeighbors

TWIN_SEARCH = Experiment(
    executors=[
        MahalanobisDistance(grouping_role=TreatmentRole()),
        FaissNearestNeighbors(grouping_role=TreatmentRole(), two_sides=True),
    ]
)
