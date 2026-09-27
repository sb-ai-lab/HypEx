"""Tests for Dataset construction, backend selection and validation."""
from __future__ import annotations

import pandas as pd
import pytest

from hypex.dataset import Dataset, FeatureRole, InfoRole, TargetRole
from hypex.utils import BackendsEnum
from hypex.utils.errors import RoleColumnError


def _df() -> pd.DataFrame:
    """Small numeric frame reused across init tests."""
    return pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0], "z": [7, 8, 9]})


def _roles() -> dict:
    """Standard roles mapping for the three-column frame."""
    return {"x": FeatureRole(), "y": TargetRole(), "z": InfoRole()}


def test_init_from_pandas_dataframe(make_dataset) -> None:
    """Dataset can be built from a pandas DataFrame."""
    ds = make_dataset(_df(), _roles())
    assert len(ds) == 3
    assert list(ds.columns) == ["x", "y", "z"]


def test_init_from_dict(make_dataset) -> None:
    """Dataset can be built from a plain dict of columns."""
    ds = make_dataset({"x": [1, 2], "y": [3, 4]}, {"x": FeatureRole(), "y": TargetRole()})
    assert len(ds) == 2
    assert set(ds.columns) == {"x", "y"}


def test_init_from_list_of_dicts(make_dataset) -> None:
    """Dataset can be built from a list of row dicts."""
    ds = make_dataset(
        [{"x": 1, "y": 10}, {"x": 2, "y": 20}],
        {"x": FeatureRole(), "y": TargetRole()},
    )
    assert len(ds) == 2
    assert set(ds.columns) == {"x", "y"}


def test_init_from_other_dataset(make_dataset) -> None:
    """Dataset can be built from another Dataset instance."""
    source = make_dataset(_df(), _roles())
    clone = Dataset(roles=None, data=source)

    assert len(clone) == len(source)
    assert list(clone.columns) == list(source.columns)


@pytest.mark.parametrize("bad_roles", [42, "roles", [1, 2]])
def test_init_raises_on_bad_roles_type(bad_roles) -> None:
    """Non-dict roles argument is rejected with an error."""
    with pytest.raises((TypeError, AttributeError)):
        Dataset(roles=bad_roles, data=_df())


@pytest.mark.parametrize("bad_data", [42, object()])
def test_init_raises_on_bad_data_type(bad_data) -> None:
    """Unsupported data types raise TypeError during backend selection."""
    with pytest.raises(TypeError):
        Dataset(roles={"x": FeatureRole()}, data=bad_data)


def test_init_raises_on_unknown_role_column() -> None:
    """A role assigned to a missing column raises RoleColumnError."""
    with pytest.raises(RoleColumnError):
        Dataset(roles={"ghost": FeatureRole()}, data=_df())


def test_default_role_fills_unspecified_columns() -> None:
    """Columns without explicit roles receive the configured default role."""
    ds = Dataset(
        roles={"x": FeatureRole()},
        data=_df(),
        default_role=InfoRole(),
    )
    assert isinstance(ds.roles["x"], FeatureRole)
    assert isinstance(ds.roles["y"], InfoRole)
    assert isinstance(ds.roles["z"], InfoRole)


def test_missing_role_without_default_raises() -> None:
    """Without default_role, unmapped columns raise KeyError."""
    with pytest.raises(KeyError):
        Dataset(roles={"x": FeatureRole()}, data=_df(), default_role=None)


def test_roles_must_be_abcrole_instances() -> None:
    """Role values that are not ABCRole instances raise TypeError."""
    with pytest.raises(TypeError):
        Dataset(roles={"x": "feature"}, data=_df())


def test_backend_selected_from_data() -> None:
    """Backend is inferred from the concrete data object type."""
    ds = Dataset(roles=_roles(), data=_df())
    assert ds.backend_type == BackendsEnum.pandas


@pytest.mark.parametrize(
    "backend", [BackendsEnum.pandas, BackendsEnum.spark], ids=["pandas", "spark"]
)
def test_backend_selected_from_enum(make_dataset, backend) -> None:
    """Explicit backend enum values are respected by the constructor."""
    if backend == BackendsEnum.spark:
        pytest.importorskip("pyspark")
    ds = make_dataset(_df(), _roles())
    assert ds.backend_type == backend


def test_create_empty(make_dataset) -> None:
    """create_empty produces a zero-row dataset that keeps its roles."""
    roles = {"a": FeatureRole(), "b": TargetRole()}
    ds = Dataset.create_empty(roles=roles)

    assert len(ds) == 0
    assert ds.is_empty()
    assert set(ds.roles.keys()) == {"a", "b"}


def test_spark_dataset_requires_session() -> None:
    """Building a SparkDataset without a session raises TypeError."""
    spark_module = pytest.importorskip("pyspark")
    from hypex.dataset.backends import SparkDataset

    with pytest.raises(TypeError):
        SparkDataset(data=pd.DataFrame({"x": [1]}), session=None)