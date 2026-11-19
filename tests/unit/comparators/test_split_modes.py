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


def test_columns_mode_is_implemented(ds) -> None:
    baseline, compared = GroupsComparator._split_data_to_buckets(
        compare_by="columns",
        target_fields_data=ds[["y"]],
        baseline_field_data=ds[["pre"]],
        group_field_data=ds[["g"]],
    )
    assert len(baseline) == 1
    assert baseline[0][1].columns == ["pre"]
    assert [item[1].columns for item in compared] == [["y"]]


def test_columns_mode_compares_every_target_and_preserves_values(ds) -> None:
    baseline, compared = GroupsComparator._split_for_columns_mode(
        ds[["pre"]], ds[["y", "z"]]
    )
    assert to_pandas(baseline[0][1])["pre"].tolist() == list(range(9))
    assert [item[1].columns for item in compared] == [["y"], ["z"]]
    assert to_pandas(compared[0][1])["y"].iloc[:3].tolist() == [1.0, 2.0, 3.0]
    assert to_pandas(compared[1][1])["z"].iloc[:3].tolist() == [5.0, 6.0, 7.0]


def test_columns_in_groups_mode_pairs_baseline_and_target_per_group(ds) -> None:
    base, comp = GroupsComparator._split_for_columns_in_groups_mode(
        ds[["g"]], ds[["pre"]], ds[["y"]]
    )
    assert len(base) == 3
    assert len(comp) == 3
    assert [len(d) for _, d in comp] == [3, 2, 4]


def test_cross_mode_uses_first_group_baseline_against_other_groups(ds) -> None:
    base, comp = GroupsComparator._split_for_cross_mode(
        ds[["g"]], ds[["pre"]], ds[["y"]]
    )
    assert len(base) == 1 and len(base[0][1]) == 3
    assert [len(d) for _, d in comp] == [2, 4]


def test_grouped_split_modes_preserve_exact_values(
    frame, backend, spark_session
) -> None:
    roles = {
        "g": GroupingRole(),
        "y": TargetRole(),
        "z": TargetRole(),
        "pre": PreTargetRole(),
    }
    data = build_dataset(frame, roles, backend, spark_session)
    base, comp = GroupsComparator._split_for_columns_in_groups_mode(
        data[["g"]], data[["pre"]], data[["y"]]
    )
    assert {name for name, _ in base} == {"a", "b", "c"}
    for name, bucket in base:
        assert bucket.columns == ["pre"]
    for name, bucket in comp:
        assert bucket.columns == ["y"]
    assert sorted(to_pandas(dict(comp)["a"])["y"]) == [1.0, 2.0, 3.0]
    assert sorted(to_pandas(dict(base)["c"])["pre"]) == [5.0, 6.0, 7.0, 8.0]

    base, comp = GroupsComparator._split_for_cross_mode(
        data[["g"]], data[["pre"]], data[["y"]]
    )
    assert len(base) == 1
    assert base[0][0] == "a"
    assert base[0][1].columns == ["pre"]
    assert sorted(to_pandas(base[0][1])["pre"]) == [0.0, 1.0, 2.0]
    assert _names(comp) == ["b", "c"]
    assert sorted(to_pandas(comp[0][1])["y"]) == [10.0, 20.0]
    assert sorted(to_pandas(comp[1][1])["y"]) == [100.0, 200.0, 300.0, 400.0]


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


def test_calc_columns_mode_compares_each_target_with_baseline(ds) -> None:
    result = _Collect.calc(
        compare_by="columns",
        target_fields_data=ds[["y", "z"]],
        baseline_field_data=ds[["pre"]],
        group_field_data=ds[["g"]],
    )
    assert sorted(result) == ["columns┆y", "columns┆z"]
    y = to_pandas(result["columns┆y"]).iloc[0]
    z = to_pandas(result["columns┆z"]).iloc[0]
    assert y["mean_base"] == pytest.approx(4.0)
    assert (y["n_base"], y["n_test"]) == (9, 9)
    assert z["mean_base"] == pytest.approx(4.0)
    assert z["mean_test"] == pytest.approx(9.0)


def test_calc_columns_in_groups_mode_uses_matching_group_baselines(ds) -> None:
    result = _Collect.calc(
        compare_by="columns_in_groups",
        target_fields_data=ds[["y"]],
        baseline_field_data=ds[["pre"]],
        group_field_data=ds[["g"]],
    )
    assert sorted(result) == ["a┆y", "b┆y", "c┆y"]
    assert to_pandas(result["a┆y"]).iloc[0]["mean_base"] == pytest.approx(1.0)
    assert to_pandas(result["a┆y"]).iloc[0]["mean_test"] == pytest.approx(2.0)
    assert to_pandas(result["b┆y"]).iloc[0]["n_base"] == 2
    assert to_pandas(result["b┆y"]).iloc[0]["mean_test"] == pytest.approx(15.0)
    assert to_pandas(result["c┆y"]).iloc[0]["n_base"] == 4
    assert to_pandas(result["c┆y"]).iloc[0]["mean_test"] == pytest.approx(250.0)


def test_calc_cross_mode_uses_first_group_baseline(ds) -> None:
    result = _Collect.calc(
        compare_by="cross",
        target_fields_data=ds[["y"]],
        baseline_field_data=ds[["pre"]],
        group_field_data=ds[["g"]],
    )
    assert sorted(result) == ["b┆y", "c┆y"]
    for key, expected_mean in (("b┆y", 15.0), ("c┆y", 250.0)):
        row = to_pandas(result[key]).iloc[0]
        assert row["mean_base"] == pytest.approx(1.0)
        assert row["mean_test"] == pytest.approx(expected_mean)


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
