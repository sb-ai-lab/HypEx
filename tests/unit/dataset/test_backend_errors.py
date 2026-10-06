"""Error handling shared by (or specific to) the Pandas and Spark backends."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from hypex.config import DatasetConfig
from hypex.dataset import Dataset, FeatureRole
from hypex.utils import (
    BackendsEnum,
    BackendTypeError,
    DataTypeError,
    MergeOnError,
)

DF = pd.DataFrame({"k": [1, 2, 3], "x": [1.0, 2.0, 3.0]})


def _build(backend: BackendsEnum, session) -> Dataset:
    return Dataset(
        roles={c: FeatureRole() for c in DF.columns},
        data=DF,
        backend=backend,
        session=session,
    )


@pytest.fixture
def pandas_ds() -> Dataset:
    return _build(BackendsEnum.pandas, None)


@pytest.fixture
def spark_ds(spark_session) -> Dataset:
    return _build(BackendsEnum.spark, spark_session)


# ---------------------------------------------------------------------------
# Merge errors (both backends via the ``make_dataset`` fixture)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs", [{"on": "missing"}, {"left_on": "missing", "right_on": "k"}]
)
def test_merge_on_missing_column_raises(make_dataset, kwargs) -> None:
    ds = make_dataset(DF, {c: FeatureRole() for c in DF.columns})
    with pytest.raises(MergeOnError):
        ds.merge(ds, **kwargs)


@pytest.mark.parametrize("other", [[1, 2, 3], {"k": 1}, None, "text"])
def test_merge_with_non_dataset_raises_data_type_error(make_dataset, other) -> None:
    ds = make_dataset(DF, {c: FeatureRole() for c in DF.columns})
    with pytest.raises(DataTypeError):
        ds.merge(other, on="k")


# ---------------------------------------------------------------------------
# Cross-backend operations
# ---------------------------------------------------------------------------
@pytest.mark.spark
def test_merge_pandas_with_spark_raises_backend_type_error(pandas_ds, spark_ds) -> None:
    with pytest.raises(BackendTypeError):
        pandas_ds.merge(spark_ds, on="k")


@pytest.mark.spark
def test_merge_spark_with_pandas_raises_backend_type_error(pandas_ds, spark_ds) -> None:
    with pytest.raises(BackendTypeError):
        spark_ds.merge(pandas_ds, on="k")


# ---------------------------------------------------------------------------
# SPARK_PANDAS_CONVERSION_LIMIT
# ---------------------------------------------------------------------------
@pytest.mark.spark
def test_conversion_limit_blocks_to_pandas(spark_ds, monkeypatch) -> None:
    monkeypatch.setattr(DatasetConfig, "SPARK_PANDAS_CONVERSION_LIMIT", 2)
    with pytest.raises(ValueError, match="exceed limit 2"):
        spark_ds.backend_data.to_backend(BackendsEnum.pandas)


@pytest.mark.spark
@pytest.mark.parametrize("method", ["to_dict", "get_values"])
def test_conversion_limit_blocks_materialising_methods(
    spark_ds, monkeypatch, method
) -> None:
    monkeypatch.setattr(DatasetConfig, "SPARK_PANDAS_CONVERSION_LIMIT", 2)
    with pytest.raises(ValueError, match="exceed limit"):
        getattr(spark_ds, method)()


@pytest.mark.spark
def test_conversion_limit_boundary_is_inclusive(spark_ds, monkeypatch) -> None:
    """Exactly ``limit`` rows is allowed; ``limit + 1`` is not."""
    monkeypatch.setattr(DatasetConfig, "SPARK_PANDAS_CONVERSION_LIMIT", len(DF))
    converted = spark_ds.backend_data.to_backend(BackendsEnum.pandas)
    assert len(converted.data) == len(DF)


@pytest.mark.spark
def test_to_backend_unsupported_target_raises(spark_ds) -> None:
    with pytest.raises(ValueError, match="Unsupported target backend"):
        spark_ds.backend_data.to_backend("foo")


def test_pandas_to_backend_pandas_is_identity(pandas_ds) -> None:
    assert (
        pandas_ds.backend_data.to_backend(BackendsEnum.pandas) is pandas_ds.backend_data
    )


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------
def test_pandas_checkpoint_is_silent_noop(pandas_ds) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pandas_ds.checkpoint()
    assert len(pandas_ds) == len(DF)


@pytest.mark.spark
def test_spark_checkpoint_without_dir_warns_and_keeps_data(
    spark_ds, spark_session
) -> None:
    if spark_session.sparkContext.getCheckpointDir() is not None:
        pytest.skip("checkpoint dir already configured for this session")
    with pytest.warns(UserWarning, match="checkpoint directory is not set"):
        spark_ds.checkpoint()
    assert len(spark_ds) == len(DF)
    assert sorted(spark_ds.backend_data.data.to_pandas()["k"]) == [1, 2, 3]
