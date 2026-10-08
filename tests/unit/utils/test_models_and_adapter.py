"""Tests for hypex.utils.models and hypex.utils.adapter."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from hypex.utils import models
from hypex.utils.adapter import Adapter


def test_cupac_models_registry() -> None:
    assert isinstance(models.CUPAC_MODELS["linear"]["pandasdataset"], LinearRegression)
    assert isinstance(models.CUPAC_MODELS["ridge"]["pandasdataset"], Ridge)
    assert isinstance(models.CUPAC_MODELS["lasso"]["pandasdataset"], Lasso)
    assert all(v["polars"] is None for v in models.CUPAC_MODELS.values())
    assert ("catboost" in models.CUPAC_MODELS) == models.CATBOOST_AVAILABLE


def test_models_without_catboost(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "catboost", None)  # makes the import fail
    try:
        reloaded = importlib.reload(models)
        assert reloaded.CATBOOST_AVAILABLE is False
        assert "catboost" not in reloaded.CUPAC_MODELS
        assert set(reloaded.CUPAC_MODELS) == {"linear", "ridge", "lasso"}
    finally:
        monkeypatch.undo()
        importlib.reload(models)


class _ToArray:
    def to_array(self):
        return [1, 2]


class _ToList:
    def to_list(self):
        return ["a"]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (None, []),
        ("abc", ["abc"]),
        (3, [3]),
        (2.5, [2.5]),
        (True, [True]),
        ([1, 2], [1, 2]),
        ((1, 2), [1, 2]),
        (np.array([1, 2]), [1, 2]),
        (pd.Index(["a", "b"]), ["a", "b"]),
        (pd.Series([1, 2]), [1, 2]),
        (_ToList(), ["a"]),
        (_ToArray(), [1, 2]),
        ({"a": 1}, [{"a": 1}]),
    ],
)
def test_to_list(data, expected) -> None:
    assert Adapter.to_list(data) == expected


def test_to_list_numpy_scalar_and_zero_dim_array() -> None:
    assert Adapter.to_list(np.int64(5)) == [5]
    assert Adapter.to_list(np.float32(1.5)) == [1.5]
    assert Adapter.to_list(np.array(7)) == [7]


def test_list_to_single() -> None:
    assert Adapter.list_to_single([]) is None
    assert Adapter.list_to_single([4]) == 4
    assert Adapter.list_to_single("notalist") is None
    with pytest.raises(ValueError, match="single item"):
        Adapter.list_to_single([1, 2])
