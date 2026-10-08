"""Tests for GroupsComparator split strategies (groups, columns, cross, matched_pairs)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex.comparators.abstract import GroupsComparator
from hypex.dataset import GroupingRole, PreTargetRole, TargetRole
from hypex.utils import NoRequiredArgumentError

from ._utils import build_dataset, to_pandas


class _Collect(GroupsComparator):
    """Concrete comparator that returns the sizes/means it receives."""

    @classmethod
    def _inner_function(cls, data, test_data=None, **kwargs):
        return {
            "n_base": len(data),
            "n_test": len(test_data),
            "mean_base": float(np.mean(to_pandas(data).iloc[:, 0])),
            "mean_test": float(np.mean(to_pandas(test_data).iloc[:, 0])),
        }


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "g": ["a", "a", "a", "b", "b", "c", "c", "c", "c"],
            "y": [1.0, 2.0, 3.0, 10.0, 20.0, 100.0, 200.0, 300.0, 400.0],
            "z": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
            "pre": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )


@pytest.fixture
def ds(frame):
    roles = {
        "g": GroupingRole(),
        "y": TargetRole(),
        "z": TargetRole(),
        "pre": PreTargetRole(),
    }
    return build_dataset(frame, roles)


def _names(buckets):
    return [name for name, _ in buckets]


def test_groups_mode_first_sorted_group_is_baseline(ds) -> None:
    base, comp = GroupsComparator._split_for_groups_mode(ds[["g"]], ds[["y"]])
    assert len(base) == 1
    assert len(base[0][1]) == 3  # group "a"
    assert [len(d) for _, d in comp] == [2, 4]
    assert [str(n) for n in _names(comp)] != []


def test_groups_mode_multi_target_keeps_first_column_only(ds) -> None:
    """Direct split warns and truncates to the first target column.

    ``execute`` avoids this by passing pre-split ``grouping_data`` instead.
    """
    with pytest.warns(UserWarning, match="must have only one column"):
        base, comp = GroupsComparator._split_for_groups_mode(ds[["g"]], ds[["y", "z"]])
    assert len(base) == 1
    assert len(comp) == 2
    assert base[0][1].columns == ["y"]


def test_groups_mode_baseline_values_are_exact(ds) -> None:
    base, _ = GroupsComparator._split_for_groups_mode(ds[["g"]], ds[["y"]])
    assert sorted(to_pandas(base[0][1])["y"]) == [1.0, 2.0, 3.0]


def test_groups_mode_ignores_group_column_in_targets(ds) -> None:
    base, comp = GroupsComparator._split_for_groups_mode(ds[["g"]], ds[["g", "y"]])
    assert all("g" not in d.columns for _, d in base + comp)


def test_groups_mode_multiple_target_check_warns_for_multi_group_columns(ds) -> None:
    with pytest.warns(UserWarning, match="must have only one column"):
        GroupsComparator._split_for_groups_mode(ds[["g", "z"]], ds[["y"]])


def test_groups_mode_empty_targets_raise(ds) -> None:
    with pytest.raises(NoRequiredArgumentError):
        GroupsComparator._split_for_groups_mode(ds[["g"]], ds[[]])


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="Issue: _split_for_columns_mode is referenced but not implemented",
)
def test_columns_mode_is_implemented(ds) -> None:
    GroupsComparator._split_data_to_buckets(
        compare_by="columns",
        target_fields_data=ds[["y"]],
        baseline_field_data=ds[["pre"]],
        group_field_data=ds[["g"]],
    )


@pytest.mark.xfail(
    strict=True,
    reason="Issue: grouping column leaks into compared data in columns_in_groups mode",
)
def test_columns_in_groups_mode_pairs_baseline_and_target_per_group(ds) -> None:
    base, comp = GroupsComparator._split_for_columns_in_groups_mode(
        ds[["g"]], ds[["pre"]], ds[["y"]]
    )
    assert len(base) == 3
    assert len(comp) == 3
    assert [len(d) for _, d in comp] == [3, 2, 4]


@pytest.mark.xfail(
    strict=True,
    reason="Issue: grouping column leaks into compared data in cross mode",
)
def test_cross_mode_uses_first_group_baseline_against_other_groups(ds) -> None:
    base, comp = GroupsComparator._split_for_cross_mode(
        ds[["g"]], ds[["pre"]], ds[["y"]]
    )
    assert len(base) == 1 and len(base[0][1]) == 3
    assert [len(d) for _, d in comp] == [2, 4]


def test_matched_pairs_mode_aligns_baseline_with_matches() -> None:
    frame = pd.DataFrame(
        {
            "g": [0, 0, 0, 1, 1],
            "y": [1.0, 2.0, 3.0, 11.0, 22.0],
            "match": [-1, -1, -1, 2, 0],  # row 3 -> row 2, row 4 -> row 0
        }
    )
    roles = {"g": GroupingRole(), "y": TargetRole(), "match": PreTargetRole()}
    ds = build_dataset(frame, roles)
    base, comp = GroupsComparator._split_for_matched_pairs_mode(
        ds[["g"]], ds[["match"]], ds[["y"]]
    )
    treated = next(d for name, d in comp if name == 1)
    matched = next(d for name, d in base if name == 1)
    assert to_pandas(treated)["y"].tolist() == [11.0, 22.0]
    assert to_pandas(matched)["y"].tolist() == [3.0, 1.0]


def test_unknown_compare_by_raises(ds) -> None:
    with pytest.raises(ValueError, match="Wrong compare_by"):
        GroupsComparator._split_data_to_buckets(
            compare_by="bogus",
            target_fields_data=ds[["y"]],
            baseline_field_data=ds[["pre"]],
            group_field_data=ds[["g"]],
        )


def test_calc_groups_mode_returns_one_result_per_compared_group(ds) -> None:
    result = _Collect.calc(
        compare_by="groups",
        target_fields_data=ds[["y"]],
        baseline_field_data=None,
        group_field_data=ds[["g"]],
    )
    assert sorted(result) == ["b", "c"]
    b = to_pandas(result["b"]).iloc[0]
    c = to_pandas(result["c"]).iloc[0]
    assert (b["n_base"], b["n_test"]) == (3, 2)
    assert (b["mean_base"], b["mean_test"]) == (2.0, 15.0)
    assert (c["n_base"], c["n_test"]) == (3, 4)
    assert c["mean_test"] == 250.0


def test_calc_with_empty_group_yields_nan_row() -> None:
    from hypex.dataset import Dataset  # noqa: F401

    base = [("a", build_dataset(pd.DataFrame({"y": [1.0, 2.0]}), {"y": TargetRole()}))]
    cmp = [
        (
            "b",
            build_dataset(
                pd.DataFrame({"y": pd.Series([], dtype=float)}), {"y": TargetRole()}
            ),
        )
    ]
    result = _Collect.calc(compare_by="groups", grouping_data=(base, cmp))
    row = to_pandas(result["b"]).iloc[0]
    assert np.isnan(row["p-value"]) and np.isnan(row["statistic"])


def test_calc_requires_compare_by_or_target() -> None:
    with pytest.raises(ValueError, match="compare_by or target_fields"):
        _Collect.calc()


def test_grouping_data_split_requires_dict() -> None:
    with pytest.raises(TypeError, match="dict of strings and datasets"):
        GroupsComparator._grouping_data_split([("a", None)], "groups", ["y"])


def test_grouping_data_split_sorts_and_pops_baseline(ds) -> None:
    data = {"b": ds, "a": ds, "c": ds}
    base, comp = GroupsComparator._grouping_data_split(data, "groups", ["y"])
    assert [n for n, _ in base] == ["a"]
    assert [n for n, _ in comp] == ["b", "c"]
