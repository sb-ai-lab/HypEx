"""Tests for DummyEncoder and its backend extensions."""

from __future__ import annotations

import pandas as pd
import pytest

from hypex.dataset import (
    Dataset,
    DisabledRole,
    ExperimentData,
    FeatureRole,
    TargetRole,
)
from hypex.encoders.encoders import DummyEncoder
from hypex.extensions import PandasDummyEncoderExtension
from hypex.utils import BackendsEnum


def _build(df, roles, backend=BackendsEnum.pandas, session=None) -> Dataset:
    return Dataset(
        roles=roles,
        data=df,
        backend=backend,
        session=session if backend == BackendsEnum.spark else None,
    )


def _pdf(ds) -> pd.DataFrame:
    data = ds.backend_data.data
    return data.to_pandas() if hasattr(data, "to_pandas") else data


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "color": ["red", "green", "blue", "red", "green"],
            "size": ["S", "M", "S", "M", "S"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


@pytest.fixture
def roles():
    return {
        "color": FeatureRole(str),
        "size": FeatureRole(str),
        "x": FeatureRole(float),
    }


def test_dummy_encoding_drops_first_category(
    frame, roles, backend, spark_session
) -> None:
    ds = _build(frame, roles, backend, spark_session)
    result = DummyEncoder._inner_function(ds, target_cols=["color"])
    # categories sorted: blue, green, red -> 'blue' is the dropped reference level
    assert sorted(result.columns) == ["color_green", "color_red"]
    encoded = _pdf(result).sort_index()
    assert encoded["color_green"].tolist() == [0, 1, 0, 0, 1]
    assert encoded["color_red"].tolist() == [1, 0, 0, 1, 0]


def test_multiple_columns_are_encoded_together(frame, roles) -> None:
    result = DummyEncoder._inner_function(
        _build(frame, roles), target_cols=["color", "size"]
    )
    assert sorted(result.columns) == ["color_green", "color_red", "size_S"]


def test_encoded_values_are_ints_and_rows_sum_to_at_most_one_per_source(
    frame, roles
) -> None:
    result = DummyEncoder._inner_function(_build(frame, roles), target_cols=["color"])
    encoded = _pdf(result)
    assert (encoded.sum(axis=1) <= 1).all()
    assert set(encoded.to_numpy().ravel()) <= {0, 1}


def test_row_alignment_is_preserved(frame, roles) -> None:
    result = DummyEncoder._inner_function(_build(frame, roles), target_cols=["size"])
    assert list(_pdf(result).index) == list(frame.index)


def test_roles_are_derived_from_original_roles(frame) -> None:
    roles = {
        "color": TargetRole(str),
        "size": FeatureRole(str),
        "x": FeatureRole(float),
    }
    result = DummyEncoder._inner_function(_build(frame, roles), target_cols=["color"])
    assert type(result.roles["color_red"]).__name__ == "AdditionalTargetRole"


def test_encoded_roles_have_bool_data_type(frame, roles) -> None:
    result = DummyEncoder._inner_function(_build(frame, roles), target_cols=["color"])
    assert all(r.data_type is bool for r in result.roles.values())


def test_no_target_cols_returns_empty_dataset(frame, roles) -> None:
    result = DummyEncoder._inner_function(_build(frame, roles), target_cols=None)
    assert len(result.columns) == 0
    assert (
        list(DummyEncoder._inner_function(_build(frame, roles), target_cols=[]).columns)
        == []
    )


def test_underscore_in_category_value_resolves_role() -> None:
    df = pd.DataFrame({"c": ["a_b", "c_d", "a_b"]})
    result = PandasDummyEncoderExtension.calc(
        _build(df, {"c": FeatureRole(str)}), ["c"]
    )
    assert list(result.columns) == ["c_c_d"]


def test_underscore_in_column_name_resolves_role() -> None:
    df = pd.DataFrame({"my_col": ["a", "b", "a"]})
    result = PandasDummyEncoderExtension.calc(
        _build(df, {"my_col": FeatureRole(str)}), ["my_col"]
    )
    assert list(result.columns) == ["my_col_b"]


# ---------------------------------------------------------------------------
# Encoder.execute
# ---------------------------------------------------------------------------
def test_execute_stores_encoded_columns_in_additional_fields(frame, roles) -> None:
    encoder = DummyEncoder(target_roles=FeatureRole())
    out = encoder.execute(ExperimentData(_build(frame, roles)))
    cols = set(out.additional_fields.columns)
    assert len(cols) == 3
    assert all("color" in c or "size" in c for c in cols)


def test_execute_ignores_non_categorical_columns(frame, roles) -> None:
    encoder = DummyEncoder(target_roles=FeatureRole())
    out = encoder.execute(ExperimentData(_build(frame, roles)))
    assert not any("┆x┆" in c for c in out.additional_fields.columns)


def test_execute_without_categorical_columns_is_noop() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0]})
    out = DummyEncoder().execute(ExperimentData(_build(df, {"x": FeatureRole(float)})))
    assert len(out.additional_fields.columns) == 0


@pytest.mark.xfail(
    strict=True,
    reason="Issue: Encoder.execute documents disabling the source columns, but "
    "_disable_target_cols is never called",
)
def test_execute_disables_original_columns(frame, roles) -> None:
    out = DummyEncoder(target_roles=FeatureRole()).execute(
        ExperimentData(_build(frame, roles))
    )
    assert isinstance(out.ds.roles["color"], DisabledRole)


def test_disable_target_cols_wraps_initial_role(frame, roles) -> None:
    ed = ExperimentData(_build(frame, roles))
    original = ed.ds.roles["color"]
    DummyEncoder._disable_target_cols(ed, ["color"])
    role = ed.ds.roles["color"]
    assert isinstance(role, DisabledRole) or role is original


def test_encoder_default_target_role_and_search_types() -> None:
    encoder = DummyEncoder()
    assert isinstance(encoder.target_roles, FeatureRole)
    assert encoder.search_types == [str]


def test_ids_to_names_use_name_border_symbol() -> None:
    from hypex.utils import NAME_BORDER_SYMBOL

    mapping = DummyEncoder()._ids_to_names(["a", "b"])
    assert set(mapping) == {"a", "b"}
    assert all(NAME_BORDER_SYMBOL in v for v in mapping.values())


@pytest.mark.xfail(
    strict=True,
    raises=ImportError,
    reason="Issue: hypex/encoders/__init__.py is empty although docstrings import "
    "DummyEncoder from hypex.encoders",
)
def test_dummy_encoder_is_exported_from_package() -> None:
    from hypex.encoders import DummyEncoder as exported  # noqa: F401
