"""Tests for AASplitter and AASplitterWithStratification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex.dataset import (
    Dataset,
    ExperimentData,
    FeatureRole,
    InfoRole,
    StatisticRole,
    StratificationRole,
)
from hypex.splitters import AASplitter, AASplitterWithStratification
from hypex.utils import ID_SPLIT_SYMBOL, BackendsEnum

N = 2000


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": np.arange(N, dtype=float),
            "s": ["a", "b"] * (N // 2),
            "k": np.arange(N),
        }
    )


ROLES = {"x": FeatureRole(), "s": StratificationRole(), "k": InfoRole()}


@pytest.fixture
def ds() -> Dataset:
    return Dataset(roles=dict(ROLES), data=_frame(), backend=BackendsEnum.pandas)


def _split_series(result: Dataset) -> pd.Series:
    data = result.backend_data.data
    data = data.to_pandas() if hasattr(data, "to_pandas") else data
    return data["split"]


# ---------------------------------------------------------------------------
# AASplitter._inner_function
# ---------------------------------------------------------------------------
def test_default_split_is_roughly_half_and_covers_all_rows(ds) -> None:
    split = _split_series(AASplitter._inner_function(ds, random_state=1))
    counts = split.value_counts()
    assert set(counts.index) == {"control", "test_1"}
    assert counts.sum() == N
    assert counts["control"] / N == pytest.approx(0.5, abs=0.04)


@pytest.mark.parametrize("control_size", [0.2, 0.3, 0.7])
def test_control_size_is_respected_within_hash_noise(ds, control_size) -> None:
    split = _split_series(
        AASplitter._inner_function(ds, random_state=1, control_size=control_size)
    )
    assert (split == "control").mean() == pytest.approx(control_size, abs=0.04)


def test_split_is_reproducible_for_same_seed(ds) -> None:
    a = _split_series(AASplitter._inner_function(ds, random_state=7))
    b = _split_series(AASplitter._inner_function(ds, random_state=7))
    assert a.equals(b)


def test_different_seeds_give_different_splits(ds) -> None:
    a = _split_series(AASplitter._inner_function(ds, random_state=1))
    b = _split_series(AASplitter._inner_function(ds, random_state=2))
    assert (a.to_numpy() == b.to_numpy()).mean() < 0.7


def test_result_keeps_index_and_has_statistic_role(ds) -> None:
    result = AASplitter._inner_function(ds, random_state=1)
    assert isinstance(result.roles["split"], StatisticRole)
    assert sorted(_split_series(result).index) == list(range(N))


def test_groups_sizes_create_named_groups(ds) -> None:
    split = _split_series(
        AASplitter._inner_function(ds, random_state=1, groups_sizes=[0.2, 0.3, 0.5])
    )
    freq = split.value_counts(normalize=True)
    assert set(freq.index) == {"control", "test_1", "test_2"}
    assert freq["control"] == pytest.approx(0.2, abs=0.04)
    assert freq["test_1"] == pytest.approx(0.3, abs=0.04)
    assert freq["test_2"] == pytest.approx(0.5, abs=0.04)


def test_sample_size_labels_only_a_fraction(ds) -> None:
    split = _split_series(
        AASplitter._inner_function(ds, random_state=1, sample_size=0.5)
    )
    assert len(split) == pytest.approx(N * 0.5, abs=0.05 * N)


def test_sample_size_zero_returns_empty(ds) -> None:
    result = AASplitter._inner_function(ds, random_state=1, sample_size=0.0)
    assert len(result) == 0
    assert "split" in result.columns


# ---------------------------------------------------------------------------
# Executor identity (params_hash)
# ---------------------------------------------------------------------------
def test_default_params_hash_is_empty() -> None:
    assert AASplitter().params_hash == ""


def test_params_hash_contains_only_non_default_params() -> None:
    assert AASplitter(random_state=3).params_hash == "rs 3"
    assert AASplitter(control_size=0.3).params_hash == "cs 0.3"
    assert AASplitter(groups_sizes=[0.5, 0.5]).params_hash == "gs [0.5, 0.5]"
    assert AASplitter(control_size=0.3, random_state=3).params_hash == "cs 0.3|rs 3"


def test_params_hash_distinguishes_ids() -> None:
    assert AASplitter(random_state=1).id != AASplitter(random_state=2).id


def test_sample_size_is_not_part_of_hash() -> None:
    assert AASplitter(sample_size=0.5).id == AASplitter().id


def test_build_from_id_round_trips_all_params() -> None:
    original = AASplitter(control_size=0.3, random_state=3, groups_sizes=[0.5, 0.5])
    rebuilt = AASplitter.build_from_id(original.id)
    assert rebuilt.control_size == 0.3
    assert rebuilt.random_state == 3
    assert rebuilt.groups_sizes == [0.5, 0.5]
    assert rebuilt.id == original.id


def test_constant_key_ignores_key_changes() -> None:
    splitter = AASplitter(key="k")
    splitter.key = "other"
    assert splitter.key == "k"


def test_non_constant_key_can_change() -> None:
    splitter = AASplitter(constant_key=False, key="k")
    splitter.key = "other"
    assert splitter.key == "other"
    assert splitter.id.endswith(f"{ID_SPLIT_SYMBOL}other")


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------
def test_execute_adds_split_column_and_groups(ds) -> None:
    splitter = AASplitter(random_state=3)
    out = splitter.execute(ExperimentData(ds))
    assert splitter.id in out.additional_fields.columns
    assert set(out.groups[splitter.id]) == {"control", "test_1"}
    sizes = {k: len(v) for k, v in out.groups[splitter.id].items()}
    assert sum(sizes.values()) == N


def test_execute_save_groups_false_skips_groups(ds) -> None:
    splitter = AASplitter(random_state=3, save_groups=False)
    out = splitter.execute(ExperimentData(ds))
    assert splitter.id in out.additional_fields.columns
    assert splitter.id not in out.groups


@pytest.mark.xfail(
    strict=True,
    reason="Issue: execute() writes the split column into the input Dataset and its "
    "source DataFrame (ExperimentData does not isolate the data)",
)
def test_execute_does_not_mutate_input() -> None:
    frame = _frame()
    dataset = Dataset(roles=dict(ROLES), data=frame, backend=BackendsEnum.pandas)
    AASplitter(random_state=5).execute(ExperimentData(dataset))
    assert list(dataset.columns) == ["x", "s", "k"]
    assert list(frame.columns) == ["x", "s", "k"]


def test_execute_is_reproducible() -> None:
    ids = []
    for _ in range(2):
        ds = Dataset(roles=dict(ROLES), data=_frame(), backend=BackendsEnum.pandas)
        splitter = AASplitter(random_state=5)
        out = splitter.execute(ExperimentData(ds))
        ids.append(sorted(out.groups[splitter.id]["control"].index))
    assert ids[0] == ids[1]


# ---------------------------------------------------------------------------
# AASplitterWithStratification
# ---------------------------------------------------------------------------
def test_stratified_split_balances_each_stratum(ds) -> None:
    result = AASplitterWithStratification._inner_function(
        ds, random_state=1, grouping_fields=["s"]
    )
    split = _split_series(result)
    strata = _frame()["s"].loc[split.index]
    for stratum in ("a", "b"):
        share = (split[strata == stratum] == "control").mean()
        assert share == pytest.approx(0.5, abs=0.06)
    assert len(split) == N


def test_stratified_without_fields_falls_back_to_plain_split(ds) -> None:
    plain = _split_series(AASplitter._inner_function(ds, random_state=1))
    strat = _split_series(
        AASplitterWithStratification._inner_function(
            ds, random_state=1, grouping_fields=None
        )
    )
    assert plain.equals(strat)


def test_stratified_execute_uses_stratification_role(ds) -> None:
    splitter = AASplitterWithStratification(random_state=3)
    out = splitter.execute(ExperimentData(ds))
    assert splitter.id in out.additional_fields.columns
    assert splitter.id.startswith("AASplitterWithStratification")


# ---------------------------------------------------------------------------
# Missing group keys in _set_value / const-group column
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "missing", [None, float("nan"), pd.NA], ids=["None", "nan", "NA"]
)
def test_set_value_skips_missing_group_keys(ds, missing, monkeypatch) -> None:
    """Group keys that are None / NaN / pd.NA are skipped (``bool(pd.NA)`` raises)."""
    splitter = AASplitter(random_state=1)
    labels = ["control", missing, "test_1"] + ["control"] * (N - 3)
    value = Dataset(
        roles={"split": StatisticRole()},
        data=pd.DataFrame({"split": labels}),
        backend=BackendsEnum.pandas,
    )

    class _Unique:
        # The backend normally turns pd.NA into None in unique(); feed the raw key.
        def __getitem__(self, _col):
            return self

        def to_dict(self):
            return dict(enumerate(["control", missing, "test_1"]))

    monkeypatch.setattr(Dataset, "unique", lambda self, *a, **k: _Unique())
    out = splitter._set_value(ExperimentData(ds), value)
    assert set(out.groups[splitter.id]) == {"control", "test_1"}


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA], ids=["None", "nan", "NA"])
def test_set_value_missing_keys_end_to_end(missing) -> None:
    frame = _frame()
    dataset = Dataset(roles=dict(ROLES), data=frame, backend=BackendsEnum.pandas)
    splitter = AASplitter(random_state=1)
    labels = pd.Series(["control", "test_1"] * (N // 2), dtype=object)
    labels.iloc[:3] = missing
    value = Dataset(
        roles={"split": StatisticRole()},
        data=pd.DataFrame({"split": labels}),
        backend=BackendsEnum.pandas,
    )
    out = splitter._set_value(ExperimentData(dataset), value)
    assert set(out.groups[splitter.id]) == {"control", "test_1"}
    assert sum(len(v) for v in out.groups[splitter.id].values()) == N - 3


@pytest.mark.parametrize("missing", [None, np.nan], ids=["None", "nan"])
def test_const_group_plan_treats_missing_label_as_free(missing) -> None:
    """None / float NaN labels go through the ``math.isnan`` guard as free rows."""
    data = Dataset(
        roles={"c": InfoRole()},
        data=pd.DataFrame(
            {"c": pd.Series(["control", "test", missing, missing], dtype=object)}
        ),
        backend=BackendsEnum.pandas,
    )
    translation, free_size, _ = AASplitter._const_group_plan(
        data=data,
        const_group_field="c",
        label_map={0: "control", 1: "test"},
        control_size=0.5,
        sample_size=1.0,
    )
    assert list(translation.values()).count("__hypex_free_const_group__") == 1
    assert free_size == 2
