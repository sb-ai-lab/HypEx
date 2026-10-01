"""Tests for TypeCaster."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex.dataset import FeatureRole, TargetRole
from hypex.transformers import TypeCaster

from ._utils import make_ds, make_ed, to_pandas


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"i": [1, 2, 3], "f": [1.5, 2.5, 3.5], "s": ["1", "2", "3"]})


@pytest.fixture
def roles():
    return {"i": FeatureRole(int), "f": FeatureRole(float), "s": FeatureRole(str)}


def test_cast_by_column_name(frame, roles) -> None:
    out = TypeCaster.calc(make_ds(frame, roles), {"s": int}, downcasting=False)
    assert to_pandas(out)["s"].tolist() == [1, 2, 3]
    assert out.roles["s"].data_type is int


def test_cast_by_source_type_casts_all_columns_with_that_type(frame, roles) -> None:
    out = TypeCaster.calc(make_ds(frame, roles), {float: int}, downcasting=False)
    assert to_pandas(out)["f"].tolist() == [1, 2, 3]
    assert out.roles["f"].data_type is int


def test_cast_restricted_by_roles(frame) -> None:
    roles = {"i": FeatureRole(int), "f": TargetRole(float), "s": FeatureRole(str)}
    out = TypeCaster.calc(
        make_ds(frame, roles), {float: int, "s": int}, roles=TargetRole(), downcasting=False
    )
    result = to_pandas(out)
    assert result["f"].tolist() == [1, 2, 3]
    assert result["s"].tolist() == ["1", "2", "3"]  # not a TargetRole -> untouched


def test_downcast_converts_float_roles_to_float32_range(frame, roles) -> None:
    out = TypeCaster.calc(make_ds(frame, roles), {}, downcasting=True)
    assert out.roles["f"].data_type is float
    assert to_pandas(out)["f"].tolist() == [1.5, 2.5, 3.5]


def test_unknown_column_raises(frame, roles) -> None:
    with pytest.raises(KeyError):
        TypeCaster.calc(make_ds(frame, roles), {"missing": int}, downcasting=False)


def test_invalid_value_cast_raises(roles) -> None:
    df = pd.DataFrame({"i": [1, 2], "f": [1.0, 2.0], "s": ["a", "b"]})
    with pytest.raises(Exception):
        TypeCaster.calc(make_ds(df, roles), {"s": int}, downcasting=False)


def test_cast_does_not_change_other_columns(frame, roles) -> None:
    out = TypeCaster.calc(make_ds(frame, roles), {"s": int}, downcasting=False)
    np.testing.assert_array_equal(to_pandas(out)["i"].to_numpy(), frame["i"].to_numpy())


def test_execute_default_roles_is_feature_role(frame, roles) -> None:
    out = TypeCaster(dtype={"s": int}, downcasting=False).execute(make_ed(frame, roles))
    assert to_pandas(out.ds)["s"].tolist() == [1, 2, 3]


def test_cast_matches_across_backends(frame, roles, backend, spark_session) -> None:
    out = TypeCaster.calc(make_ds(frame, roles, backend, spark_session), {"s": int}, downcasting=False)
    assert sorted(to_pandas(out)["s"].tolist()) == [1, 2, 3]


def test_is_transformer() -> None:
    assert TypeCaster(dtype={})._is_transformer is True
