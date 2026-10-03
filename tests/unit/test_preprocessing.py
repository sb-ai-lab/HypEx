"""Tests for the PREPROCESSING_DATA pipeline definition."""

from __future__ import annotations

from hypex.encoders.encoders import DummyEncoder
from hypex.experiments.base import Experiment
from hypex.preprocessing import PREPROCESSING_DATA
from hypex.transformers.category_agg import CategoryAggregator
from hypex.transformers.filters import (
    ConstFilter,
    CorrFilter,
    CVFilter,
    NanFilter,
    OutliersFilter,
)
from hypex.transformers.na_filler import NaFiller


def test_preprocessing_pipeline_structure() -> None:
    assert isinstance(PREPROCESSING_DATA, Experiment)
    assert [type(e) for e in PREPROCESSING_DATA.executors] == [
        NaFiller,
        CategoryAggregator,
        CorrFilter,
        CVFilter,
        NanFilter,
        ConstFilter,
        OutliersFilter,
        DummyEncoder,
    ]


def test_preprocessing_pipeline_parameters() -> None:
    filler = PREPROCESSING_DATA.executors[0]
    outliers = PREPROCESSING_DATA.executors[6]
    assert filler.method == "ffill"
    assert outliers.lower_percentile == 0.05
    assert outliers.upper_percentile == 0.95
