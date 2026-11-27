"""Pandas compatibility checks that do not require a Spark session."""

from __future__ import annotations

import pandas as pd
import pytest

from hypex.dataset import Dataset, FeatureRole, GroupingRole


def _dataset():
    frame = pd.DataFrame(
        {"group": pd.Series(["a", "a", "b"], dtype=object), "value": [1, 2, 3]}
    )
    frame.index.name = "row_id"
    return Dataset(data=frame, roles={"group": GroupingRole(), "value": FeatureRole()})


def test_set_index_returns_new_dataset_without_mutating_source():
    source = _dataset()
    original = source.raw_data.copy(deep=True)

    result = source.set_index("group")

    pd.testing.assert_frame_equal(source.raw_data, original)
    assert result.index.tolist() == ["a", "a", "b"]
    assert list(result.columns) == ["value"]


@pytest.mark.parametrize("selector", ["object", object, ["object"], [object]])
def test_object_selector_includes_and_excludes_string_columns(selector):
    source = _dataset()

    assert list(source.select_dtypes(include=selector).columns) == ["group"]
    assert list(source.select_dtypes(exclude=selector).columns) == ["value"]


@pytest.mark.parametrize("use_iterator", [False, True])
def test_reindex_fills_strings_with_numeric_value_and_preserves_index_name(
    use_iterator,
):
    source = _dataset()
    labels = iter([0, 1, 9]) if use_iterator else [0, 1, 9]

    result = source.reindex(labels, fill_value=-1)

    assert result.raw_data.loc[9, "group"] == -1
    assert result.raw_data.loc[9, "value"] == -1
    assert result.index.name == "row_id"
    assert source.raw_data["group"].tolist() == ["a", "a", "b"]


def test_reindex_existing_labels_preserves_string_dtype():
    source = _dataset()

    result = source.reindex([0, 1], fill_value=-1)

    pd.testing.assert_frame_equal(result.raw_data, source.raw_data.iloc[:2])


def test_groupby_apply_receives_grouping_columns():
    result = (
        _dataset()
        .groupby("group")
        .apply(
            lambda part: pd.Series(
                {"group_seen": part["group"].iloc[0], "total": part["value"].sum()}
            )
        )
    )

    frame = result.raw_data.sort_index()
    assert frame["group_seen"].tolist() == ["a", "b"]
    assert frame["total"].tolist() == [3, 3]
