"""Additional GroupedDataset tests: reducers with columns, list agg, apply, iteration."""

from __future__ import annotations

import pandas as pd
import pytest

from hypex.dataset import Dataset, FeatureRole, GroupingRole, InfoRole, StatisticRole
from hypex.dataset.groupby_dataset import GroupedDataset
from hypex.utils import NAME_BORDER_SYMBOL, BackendsEnum


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "v": [1.0, 3.0, 10.0, 30.0],
            "w": [2.0, 4.0, 20.0, 40.0],
        }
    )


def _roles() -> dict:
    return {"g": GroupingRole(), "v": FeatureRole(), "w": FeatureRole()}


def _grouped(make_dataset):
    return make_dataset(_df(), _roles()).groupby("g")


def _pdf(ds) -> pd.DataFrame:
    data = ds.backend_data.data
    return data.to_pandas() if hasattr(data, "to_pandas") else data


# ---------------------------------------------------------------------------
# Reducers (values, not just lengths)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "reducer,expected",
    [
        ("sum", [4.0, 40.0]),
        ("mean", [2.0, 20.0]),
        ("min", [1.0, 10.0]),
        ("max", [3.0, 30.0]),
        ("median", [2.0, 20.0]),
        ("prod", [3.0, 300.0]),
        ("first", [1.0, 10.0]),
        ("last", [3.0, 30.0]),
    ],
)
def test_reducer_values_with_column_subset(
    make_dataset, backend, reducer, expected
) -> None:
    if backend == BackendsEnum.spark and reducer in {"prod", "first", "last", "median"}:
        pytest.skip("not every reducer name is mapped on Spark")
    result = getattr(_grouped(make_dataset), reducer)("v")
    out = _pdf(result)
    assert list(result.columns) == ["v"]
    assert sorted(out["v"].tolist()) == sorted(expected)


@pytest.mark.parametrize("reducer", ["std", "var"])
def test_std_var_with_column_subset(make_dataset, backend, reducer) -> None:
    if backend == BackendsEnum.spark and reducer == "var":
        pytest.skip("Spark var mapping issue")
    out = _pdf(getattr(_grouped(make_dataset), reducer)("v"))
    expected = _df().groupby("g")["v"].agg(reducer)
    assert sorted(out["v"].tolist()) == pytest.approx(sorted(expected.tolist()))


def test_reducer_keeps_roles_of_original_columns(make_dataset) -> None:
    result = _grouped(make_dataset).mean()
    assert isinstance(result.roles["v"], FeatureRole)


def test_agg_list_builds_multi_level_names(make_dataset, backend) -> None:
    if backend == BackendsEnum.spark:
        pytest.skip("list agg naming is pandas specific")
    result = _grouped(make_dataset).agg(["mean", "max"])
    sep = NAME_BORDER_SYMBOL
    assert set(result.columns) == {
        f"{c}{sep}{s}" for c in ("v", "w") for s in ("mean", "max")
    }
    assert all(isinstance(result.roles[c], StatisticRole) for c in result.columns)
    assert _pdf(result)[f"v{sep}max"].tolist() == [3.0, 30.0]


def test_agg_dict(make_dataset, backend) -> None:
    if backend == BackendsEnum.spark:
        pytest.skip("dict agg is pandas specific")
    result = _grouped(make_dataset).agg({"v": "sum", "w": "max"})
    out = _pdf(result)
    assert out["v"].tolist() == [4.0, 40.0]
    assert out["w"].tolist() == [4.0, 40.0]


def test_agg_drops_fully_null_columns(make_dataset, backend) -> None:
    if backend == BackendsEnum.spark:
        pytest.skip("null-column dropping checked on pandas")
    df = _df()
    df["n"] = float("nan")
    ds = make_dataset(df, {**_roles(), "n": FeatureRole()})
    result = ds.groupby("g").mean()
    assert "n" not in result.columns
    assert {"v", "w"} <= set(result.columns)


def test_agg_value_counts_alone_delegates(make_dataset) -> None:
    df = pd.DataFrame({"g": ["a", "a", "b"], "c": ["x", "y", "y"]})
    ds = make_dataset(df, {"g": GroupingRole(), "c": FeatureRole()})
    result = ds.groupby("g").agg("value_counts")
    assert list(result.columns) == ["c"]


def test_agg_list_with_value_counts_adds_suffixed_columns(
    make_dataset, backend
) -> None:
    if backend == BackendsEnum.spark:
        pytest.skip("combined agg checked on pandas")
    df = pd.DataFrame(
        {"g": ["a", "a", "b"], "c": ["x", "y", "y"], "v": [1.0, 2.0, 3.0]}
    )
    ds = make_dataset(df, {"g": GroupingRole(), "c": FeatureRole(), "v": FeatureRole()})
    result = ds.groupby("g").agg(["value_counts"])
    assert f"c{NAME_BORDER_SYMBOL}value_counts" in result.columns
    mixed = ds.groupby("g").agg(["max", "value_counts"])
    assert f"v{NAME_BORDER_SYMBOL}max" in mixed.columns
    assert f"c{NAME_BORDER_SYMBOL}value_counts" in mixed.columns


def test_value_counts_with_explicit_columns(make_dataset) -> None:
    df = pd.DataFrame(
        {"g": ["a", "a", "b"], "c": ["x", "y", "y"], "d": ["p", "p", "q"]}
    )
    ds = make_dataset(df, {"g": GroupingRole(), "c": FeatureRole(), "d": FeatureRole()})
    result = ds.groupby("g").value_counts("c")
    assert list(result.columns) == ["c"]


# ---------------------------------------------------------------------------
# size / apply / iteration
# ---------------------------------------------------------------------------
def test_size_returns_group_sizes(make_dataset) -> None:
    result = _grouped(make_dataset).size()
    assert list(result.columns) == ["size"]
    assert sorted(_pdf(result)["size"].tolist()) == [2, 2]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_apply_assigns_info_roles(make_dataset, backend) -> None:
    if backend == BackendsEnum.spark:
        pytest.skip("apply with arbitrary callables is pandas specific")
    result = _grouped(make_dataset).apply(lambda part: part[["v", "w"]].sum())
    assert all(isinstance(r, InfoRole) for r in result.roles.values())
    assert sorted(_pdf(result)["v"].tolist()) == [4.0, 40.0]


def test_iteration_yields_key_and_filtered_dataset(make_dataset) -> None:
    seen = {}
    for key, part in _grouped(make_dataset):
        seen[key if not isinstance(key, tuple) else key[0]] = _pdf(part)
    assert set(seen) == {"a", "b"}
    assert sorted(seen["a"]["v"].tolist()) == [1.0, 3.0]
    assert sorted(seen["b"]["v"].tolist()) == [10.0, 30.0]


def test_iteration_roles_only_for_present_columns(make_dataset) -> None:
    for _, part in _grouped(make_dataset):
        assert set(part.roles) <= {"g", "v", "w"}
        assert isinstance(part.roles["v"], FeatureRole)


def test_len_without_group_cols_is_one() -> None:
    grouped = GroupedDataset(None, Dataset, {}, {}, group_cols=None)
    assert len(grouped) == 1


# ---------------------------------------------------------------------------
# Manually constructed GroupedDataset (backend-agnostic branches)
# ---------------------------------------------------------------------------
def test_iter_without_backend_raises_type_error() -> None:
    grouped = GroupedDataset(None, Dataset, {}, {}, group_cols=["g"])
    with pytest.raises(TypeError, match="not iterable"):
        next(iter(grouped))


def test_iter_without_group_cols_raises_type_error() -> None:
    grouped = GroupedDataset(
        None, Dataset, {}, {}, group_cols=[], backend_data=object()
    )
    with pytest.raises(TypeError, match="not iterable"):
        next(iter(grouped))


def test_execute_agg_unsupported_groupby_type_raises() -> None:
    grouped = GroupedDataset(object(), Dataset, {}, {})
    with pytest.raises(TypeError, match="Unsupported groupby"):
        grouped.agg("sum")


def test_apply_unsupported_groupby_type_raises() -> None:
    grouped = GroupedDataset(object(), Dataset, {}, {})
    with pytest.raises(NotImplementedError):
        grouped.apply(lambda x: x)


def test_list_groupby_agg_concatenates_group_results() -> None:
    parts = [("a", _df().iloc[:2][["v"]]), ("b", _df().iloc[2:][["v"]])]
    grouped = GroupedDataset(parts, Dataset, {"v": FeatureRole()}, {})
    result = grouped.agg("sum")
    assert sorted(_pdf(result).iloc[:, 0].tolist()) == [4.0, 40.0]


def test_list_groupby_empty_agg_returns_empty_dataset() -> None:
    grouped = GroupedDataset([], Dataset, {}, {})
    result = grouped.agg("sum")
    assert len(result) == 0


def test_list_groupby_apply_concatenates_group_results() -> None:
    parts = [("a", _df().iloc[:2][["v"]]), ("b", _df().iloc[2:][["v"]])]
    grouped = GroupedDataset(parts, Dataset, {"v": FeatureRole()}, {})
    result = grouped.apply(lambda col: col * 2)
    assert sorted(_pdf(result)["v"].tolist()) == [2.0, 6.0, 20.0, 60.0]
    assert all(isinstance(r, InfoRole) for r in result.roles.values())


def test_list_groupby_empty_apply_returns_none() -> None:
    assert GroupedDataset([], Dataset, {}, {}).apply(lambda x: x) is None


def test_get_agg_roles_copies_known_and_defaults_unknown() -> None:
    role = FeatureRole()
    grouped = GroupedDataset(None, Dataset, {"v": role}, {})
    roles = grouped._get_agg_roles(["v", "other"])
    assert isinstance(roles["v"], FeatureRole) and roles["v"] is not role
    assert isinstance(roles["other"], StatisticRole)
