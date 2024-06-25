from hypex.experiments import Experiment
from hypex.transformers.filters import (
    NanFilter,
    CorrFilter,
    ConstFilter,
    CVFilter,
    OutliersFilter,
)
from hypex.transformers.category_agg import CategoryAggregator
from hypex.encoders.encoders import DummyEncoder

PREPROCESSING_DATA = Experiment(
    executors=[
        CategoryAggregator(),
        CorrFilter(),
        CVFilter(),
        NanFilter(),
        ConstFilter(),
        OutliersFilter(lower_percentile=0.05, upper_percentile=0.95),
        DummyEncoder(),
    ]
)
