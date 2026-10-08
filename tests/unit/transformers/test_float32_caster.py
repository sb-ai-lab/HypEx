"""Tests for Float32Caster."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex.dataset import FeatureRole, InfoRole, TargetRole, TreatmentRole
from hypex.transformers import Float32Caster

from ._utils import make_ds, make_ed, to_pandas


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f": [0.1, 0.2, 0.3],
            "t": [1.1, 2.2, 3.3],
            "i": [1, 2, 3],
            "info": [0.5, 0.6, 0.7],
        }
    )


@pytest.fixture
def roles():
    return {
        "f": FeatureRole(float),
        "t": TargetRole(float),
        "i": FeatureRole(int),
        "info": InfoRole(float),
    }


def test_defaults() -> None:
    caster = Float32Caster()
    assert [type(r) for r in caster.target_roles] == [FeatureRole, TargetRole]
    assert caster.columns is None
    assert caster.search_types == [float]


def test_single_role_and_column_are_listified() -> None:
    caster = Float32Caster(target_roles=TargetRole(), columns="f")
    assert len(caster.target_roles) == 1
    assert caster.columns == ["f"]


def test_execute_casts_feature_and_target_float_columns_only(frame, roles) -> None:
    out = Float32Caster().execute(make_ed(frame, roles))
    df = to_pandas(out.ds)
    assert df["f"].dtype == np.float32
    assert df["t"].dtype == np.float32
    assert df["i"].dtype == frame["i"].dtype
    assert df["info"].dtype == np.float64
    np.testing.assert_allclose(df["f"], frame["f"], rtol=1e-6)


def test_roles_data_type_restored_to_float(frame, roles) -> None:
    out = Float32Caster().execute(make_ed(frame, roles))
    assert out.ds.roles["f"].data_type is float
    assert out.ds.roles["t"].data_type is float
    assert out.ds.roles["i"].data_type is int


def test_execute_with_explicit_columns_ignores_roles(frame, roles) -> None:
    out = Float32Caster(columns=["info", "missing"]).execute(make_ed(frame, roles))
    df = to_pandas(out.ds)
    assert df["info"].dtype == np.float32
    assert df["f"].dtype == np.float64


def test_execute_with_custom_roles(frame, roles) -> None:
    out = Float32Caster(target_roles=InfoRole()).execute(make_ed(frame, roles))
    df = to_pandas(out.ds)
    assert df["info"].dtype == np.float32
    assert df["f"].dtype == np.float64


def test_execute_respects_data_type_of_requested_role(frame, roles) -> None:
    ed = make_ed(frame, roles)
    # the only Info column is float, so a request for int Info columns casts nothing
    assert Float32Caster(target_roles=InfoRole(int)).execute(ed) is ed
    out = Float32Caster(target_roles=InfoRole(float)).execute(ed)
    df = to_pandas(out.ds)
    assert df["info"].dtype == np.float32
    assert df["f"].dtype == np.float64


def test_execute_returns_input_when_nothing_to_cast(frame, roles) -> None:
    ed = make_ed(frame, roles)
    assert Float32Caster(columns=["nope"]).execute(ed) is ed
    assert Float32Caster(target_roles=TreatmentRole()).execute(ed) is ed


def test_calc_without_target_cols_casts_all_float_columns(frame, roles) -> None:
    out = Float32Caster.calc(make_ds(frame, roles))
    df = to_pandas(out)
    assert {c for c in df if df[c].dtype == np.float32} == {"f", "t", "info"}
    assert df["i"].dtype != np.float32


def test_calc_with_explicit_target_cols(frame, roles) -> None:
    out = Float32Caster.calc(make_ds(frame, roles), target_cols=["t"])
    df = to_pandas(out)
    assert df["t"].dtype == np.float32
    assert df["f"].dtype == np.float64


def test_inner_function_empty_target_cols_returns_same_dataset(frame, roles) -> None:
    ds = make_ds(frame, roles)
    assert Float32Caster._inner_function(ds, []) is ds
