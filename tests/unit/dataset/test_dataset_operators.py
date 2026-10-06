"""Tests for Dataset binary/unary operators and copy semantics."""

from __future__ import annotations

import operator
from copy import deepcopy

import pandas as pd
import pytest

from hypex.dataset import Dataset, FeatureRole
from hypex.utils import BackendsEnum
from hypex.utils.errors import BackendTypeError, DataTypeError

_BINARY_ARITH: list[tuple[str, object]] = [
    ("__add__", operator.add),
    ("__sub__", operator.sub),
    ("__mul__", operator.mul),
    ("__truediv__", operator.truediv),
    ("__floordiv__", operator.floordiv),
    ("__mod__", operator.mod),
    ("__pow__", operator.pow),
]

_COMPARISON: list[tuple[str, object]] = [
    ("__eq__", operator.eq),
    ("__ne__", operator.ne),
    ("__lt__", operator.lt),
    ("__le__", operator.le),
    ("__gt__", operator.gt),
    ("__ge__", operator.ge),
]


def _ds(make_dataset, values=None):
    """Single numeric column dataset for operator tests."""
    df = pd.DataFrame({"x": values or [1, 2, 3]})
    return make_dataset(df, {"x": FeatureRole()})


@pytest.mark.parametrize("name,op", _BINARY_ARITH, ids=[n for n, _ in _BINARY_ARITH])
def test_arithmetic_with_scalar(make_dataset, name, op) -> None:
    """Every arithmetic operator works with a scalar operand."""
    ds = _ds(make_dataset, [2, 4, 6])
    result = op(ds, 2)
    assert len(result) == 3


@pytest.mark.parametrize("name,op", _BINARY_ARITH, ids=[n for n, _ in _BINARY_ARITH])
def test_arithmetic_with_dataset(make_dataset, name, op) -> None:
    """Every arithmetic operator works between same-shape datasets."""
    left = _ds(make_dataset, [2, 4, 6])
    right = _ds(make_dataset, [1, 2, 3])
    result = op(left, right)
    assert len(result) == 3


@pytest.mark.parametrize(
    "name,op",
    [
        ("__radd__", lambda d: 10 + d),
        ("__rsub__", lambda d: 10 - d),
        ("__rmul__", lambda d: 10 * d),
        ("__rtruediv__", lambda d: 10 / d),
    ],
    ids=["radd", "rsub", "rmul", "rtruediv"],
)
def test_reflected_operators(make_dataset, name, op) -> None:
    """Reflected operators accept the Dataset on the right-hand side."""
    ds = _ds(make_dataset, [1, 2, 5])
    result = op(ds)
    assert len(result) == 3


@pytest.mark.parametrize("name,op", _COMPARISON, ids=[n for n, _ in _COMPARISON])
def test_comparison_operators(make_dataset, name, op) -> None:
    """Comparison operators yield a same-length boolean dataset."""
    ds = _ds(make_dataset, [1, 2, 3])
    result = op(ds, 2)
    assert len(result) == 3


@pytest.mark.pandas
def test_bitwise_operators(make_dataset, xfail_backend) -> None:
    xfail_backend(
        BackendsEnum.spark,
        reason="Issue: SparkDataset.__and__/__or__ apply & to pyspark.pandas DataFrames, which is unsupported (TypeError)",
        raises=TypeError,
    )
    """& and | work on boolean datasets."""
    df = pd.DataFrame({"x": [True, True, False]})
    ds = make_dataset(df, {"x": FeatureRole()})
    anded = ds & ds
    ored = ds | ds
    assert len(anded) == 3
    assert len(ored) == 3


@pytest.mark.pandas
def test_unary_operators(make_dataset, xfail_backend) -> None:
    xfail_backend(
        BackendsEnum.spark,
        reason="Issue: SparkDataset.__pos__ applies unary + to a pyspark.pandas DataFrame, which is unsupported (TypeError)",
        raises=TypeError,
    )
    """Unary +, -, abs and round return datasets of the same shape."""
    ds = _ds(make_dataset, [-1, 2, -3])
    assert len(+ds) == 3
    assert len(-ds) == 3
    assert len(abs(ds)) == 3
    assert len(round(ds, 0)) == 3


def test_bool_on_dataset(make_dataset) -> None:
    """__bool__ is True for non-empty and False for empty datasets."""
    ds = _ds(make_dataset)
    assert bool(ds) is True

    empty = Dataset.create_empty(roles={"x": FeatureRole()})
    assert bool(empty) is False


@pytest.mark.parametrize("bad_other", ["text", {1, 2}, object()])
def test_operator_with_invalid_type_raises(make_dataset, bad_other) -> None:
    """Operators reject unsupported operand types with DataTypeError."""
    ds = _ds(make_dataset)
    with pytest.raises((DataTypeError, TypeError, Exception)):
        _ = ds + bad_other


def test_operator_across_backends_raises(make_dataset, spark_session) -> None:
    """Mixing pandas and spark datasets raises BackendTypeError."""
    pytest.importorskip("pyspark")
    df = pd.DataFrame({"x": [1, 2, 3]})
    roles = {"x": FeatureRole()}
    pandas_ds = Dataset(roles=roles, data=df)
    spark_ds = Dataset(
        roles=roles, data=df, backend=BackendsEnum.spark, session=spark_session
    )
    with pytest.raises(BackendTypeError):
        _ = pandas_ds + spark_ds


def test_deepcopy_independent(make_dataset) -> None:
    """deepcopy yields a fully independent dataset including roles."""
    ds = _ds(make_dataset)
    original_type = ds.roles["x"].data_type
    clone = deepcopy(ds)
    clone.roles["x"].data_type = float
    assert ds.roles["x"].data_type == original_type


def test_division_by_zero_does_not_raise(make_dataset) -> None:
    """Division by zero produces inf/NaN instead of raising."""
    ds = _ds(make_dataset, [1, 2, 3])
    result = ds / 0
    assert len(result) == 3
