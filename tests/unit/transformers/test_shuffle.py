"""Tests for Shuffle."""

from __future__ import annotations

import pandas as pd
import pytest

from hypex.dataset import FeatureRole
from hypex.transformers import Shuffle
from hypex.utils import BackendsEnum

from ._utils import make_ds, to_pandas


@pytest.fixture
def ds():
    df = pd.DataFrame({"x": list(range(50)), "y": [float(i) for i in range(50)]})
    return make_ds(df, {"x": FeatureRole(), "y": FeatureRole()})


def test_shuffle_is_permutation_and_reproducible(ds) -> None:
    first = to_pandas(Shuffle._inner_function(ds, random_state=1))
    second = to_pandas(Shuffle._inner_function(ds, random_state=1))
    assert sorted(first["x"]) == list(range(50))
    assert first["x"].tolist() == second["x"].tolist()
    assert first["x"].tolist() != list(range(50))


def test_shuffle_execute(ds) -> None:
    from hypex.dataset import ExperimentData

    out = Shuffle(random_state=3).execute(ExperimentData(ds))
    assert len(out.ds) == 50


def test_shuffle_id_depends_on_random_state() -> None:
    assert Shuffle(random_state=1).id != Shuffle(random_state=2).id


def test_shuffle_stores_random_state() -> None:
    assert Shuffle(random_state=7).random_state == 7


def test_shuffle_on_spark(spark_session) -> None:
    df = pd.DataFrame({"x": list(range(50)), "y": [float(i) for i in range(50)]})
    ds = make_ds(
        df,
        {"x": FeatureRole(), "y": FeatureRole()},
        backend=BackendsEnum.spark,
        session=spark_session,
    )
    first = to_pandas(Shuffle._inner_function(ds, random_state=1))
    second = to_pandas(Shuffle._inner_function(ds, random_state=1))
    assert sorted(first["x"]) == list(range(50))
    assert first["x"].tolist() == second["x"].tolist()
