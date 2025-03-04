from .encoders.encoders import DummyEncoder
from .experiments.base import Experiment
from .transformers.category_agg import CategoryAggregator
from .transformers.filters import (
    ConstFilter,
    CorrFilter,
    CVFilter,
    NanFilter,
    OutliersFilter,
)
from .transformers.na_filler import NaFiller

PREPROCESSING_DATA = Experiment(
    executors=[
        NaFiller(method="ffill"),
        CategoryAggregator(),
        CorrFilter(),
        CVFilter(),
        NanFilter(),
        ConstFilter(),
        OutliersFilter(lower_percentile=0.05, upper_percentile=0.95),
        DummyEncoder(),
    ]
)
