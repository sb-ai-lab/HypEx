"""Tests for CategoryAggregator."""

from __future__ import annotations

import pandas as pd
import pytest

from hypex.dataset import FeatureRole, TargetRole
from hypex.transformers.category_agg import CategoryAggregator

from ._utils import make_ds, make_ed, to_pandas


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cat": ["a"] * 5 + ["b"] * 4 + ["c", "d"],
            "other": ["x"] * 6 + ["y"] * 5,
            "num": range(11),
        }
    )


@pytest.fixture
def roles():
    return {
        "cat": FeatureRole(str),
        "other": FeatureRole(str),
        "num": TargetRole(int),
    }


def test_defaults() -> None:
    agg = CategoryAggregator()
    assert agg.threshold == 15
    assert isinstance(agg.target_roles, FeatureRole)
    assert agg.search_types == [str]


def test_rare_categories_are_replaced(frame, roles) -> None:
    out = CategoryAggregator._inner_function(
        make_ds(frame, roles), ["cat"], threshold=3, new_group_name="rare"
    )
    assert to_pandas(out)["cat"].tolist() == ["a"] * 5 + ["b"] * 4 + ["rare"] * 2


def test_frequent_categories_untouched(frame, roles) -> None:
    out = CategoryAggregator._inner_function(
        make_ds(frame, roles), ["cat", "other"], threshold=2, new_group_name="rare"
    )
    df = to_pandas(out)
    assert df["other"].tolist() == frame["other"].tolist()
    assert df["cat"].tolist() == ["a"] * 5 + ["b"] * 4 + ["rare"] * 2


def test_execute_aggregates_only_str_feature_columns(frame, roles) -> None:
    agg = CategoryAggregator(threshold=5, new_group_name="rare")
    out = agg.execute(make_ed(frame, roles))
    df = to_pandas(out.ds)
    assert df["cat"].tolist() == ["a"] * 5 + ["rare"] * 6
    assert df["other"].tolist() == frame["other"].tolist()  # counts 6 and 5, not < 5
    assert df["num"].tolist() == list(range(11))
