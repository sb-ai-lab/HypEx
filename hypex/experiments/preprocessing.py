import copy

from hypex.dataset import Dataset, ExperimentData, InfoRole, TreatmentRole, TargetRole, FeatureRole
from hypex.experiments import Experiment
from hypex.transformers.filters import NanFilter, CorrFilter, ConstFilter, CVFilter, OutliersFilter
from hypex.transformers.category_agg import CategoryAggregator
from hypex.encoders.encoders import DummyEncoder

PREPROCESSING_DATA = Experiment(
    executors=[
        CategoryAggregator(),
        CVFilter(),
        NanFilter(),
        CorrFilter(),
        ConstFilter(),
        OutliersFilter(lower_percentile=0.05, upper_percentile=0.95),
        DummyEncoder(),
    ]
)