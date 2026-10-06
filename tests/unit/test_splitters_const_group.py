"""AASplitter with a ConstGroupRole column (pinned control/test rows)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex.dataset import (
    ConstGroupRole,
    Dataset,
    ExperimentData,
    FeatureRole,
    StratificationRole,
)
from hypex.splitters import AASplitter, AASplitterWithStratification
from hypex.utils import BackendsEnum

N = 2000


def _frame(
    pinned_control: int = 100, pinned_test: int = 0, seed: int = 0
) -> pd.DataFrame:
    labels = np.array([None] * N, dtype=object)
    labels[:pinned_control] = "control"
    labels[pinned_control : pinned_control + pinned_test] = "test"
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "x": rng.normal(size=N),
            "s": ["a", "b"] * (N // 2),
            "c": labels,
        }
    )


def _dataset(df: pd.DataFrame, backend=BackendsEnum.pandas, session=None) -> Dataset:
    return Dataset(
        roles={"x": FeatureRole(), "s": StratificationRole(), "c": ConstGroupRole()},
        data=df,
        backend=backend,
        session=session,
    )


def _split(result: Dataset) -> pd.Series:
    data = result.backend_data.data
    data = data.to_pandas() if hasattr(data, "to_pandas") else data
    return data["split"]


def _run(df, **kwargs) -> pd.Series:
    kwargs.setdefault("random_state", 3)
    out = AASplitter._inner_function(_dataset(df), const_group_field="c", **kwargs)
    return _split(out).sort_index()


def test_pinned_control_rows_always_land_in_control() -> None:
    df = _frame(pinned_control=150)
    split = _run(df)
    assert len(split) == N
    assert (split.iloc[:150] == "control").all()


def test_total_control_share_matches_requested_size() -> None:
    split = _run(_frame(pinned_control=150), control_size=0.5)
    assert (split == "control").mean() == pytest.approx(0.5, abs=0.04)


def test_free_rows_are_split_between_control_and_test() -> None:
    split = _run(_frame(pinned_control=150))
    free = split.iloc[150:]
    assert set(free) == {"control", "test_1"}


def test_test_alias_pins_rows_to_first_test_group() -> None:
    df = _frame(pinned_control=50, pinned_test=80)
    split = _run(df)
    assert (split.iloc[:50] == "control").all()
    assert (split.iloc[50:130] == "test_1").all()


def test_unknown_pinned_label_raises() -> None:
    df = _frame()
    df.loc[0, "c"] = "oops"
    with pytest.raises(ValueError, match="Unknown constant group"):
        _run(df)


def test_pinned_control_over_quota_warns() -> None:
    df = _frame(pinned_control=1500)
    with pytest.warns(UserWarning, match="exceed the requested control quota"):
        split = _run(df, control_size=0.5)
    assert (split.iloc[:1500] == "control").all()
    # every free row must go to test once the quota is exceeded
    assert (split.iloc[1500:] == "test_1").all()


def test_all_rows_pinned_returns_only_pinned_labels() -> None:
    df = _frame(pinned_control=N // 2, pinned_test=N // 2)
    split = _run(df)
    assert (split.iloc[: N // 2] == "control").all()
    assert (split.iloc[N // 2 :] == "test_1").all()


def test_sample_size_samples_only_free_rows() -> None:
    df = _frame(pinned_control=200)
    split = _run(df, sample_size=0.5)
    assert (split.iloc[:200] == "control").all()
    assert len(split) < N
    assert len(split) == pytest.approx(200 + 0.5 * (N - 200), abs=0.05 * N)


def test_non_string_const_column_is_cast_to_string() -> None:
    df = _frame()
    df["c"] = pd.Series([np.nan] * N, dtype=float)
    split = _run(df)
    assert set(split) == {"control", "test_1"}
    assert (split == "control").mean() == pytest.approx(0.5, abs=0.04)


def test_groups_sizes_with_const_group_keeps_three_groups() -> None:
    df = _frame(pinned_control=100)
    split = _run(df, groups_sizes=[0.4, 0.3, 0.3])
    assert set(split) == {"control", "test_1", "test_2"}
    assert (split.iloc[:100] == "control").all()
    assert (split == "control").mean() == pytest.approx(0.4, abs=0.04)
    assert (split == "test_1").mean() == pytest.approx(0.3, abs=0.04)


def test_groups_sizes_all_control_leaves_equal_test_groups() -> None:
    df = _frame(pinned_control=100)
    split = _run(df, groups_sizes=[1.0, 0.0, 0.0])
    assert (split == "control").mean() > 0.95


def test_execute_reads_const_role_and_keeps_pinned_in_control() -> None:
    df = _frame(pinned_control=120)
    splitter = AASplitter(random_state=5, save_groups=False)
    out = splitter.execute(ExperimentData(_dataset(df)))
    column = out.additional_fields.backend_data.data[splitter.id]
    assert (column.iloc[:120] == "control").all()
    assert (column == "control").mean() == pytest.approx(0.5, abs=0.04)


def test_stratified_execute_reads_const_role() -> None:
    df = _frame(pinned_control=120)
    splitter = AASplitterWithStratification(random_state=5, save_groups=False)
    out = splitter.execute(ExperimentData(_dataset(df)))
    column = out.additional_fields.backend_data.data[splitter.id]
    assert len(column) == N
    assert (column.iloc[:120] == "control").all()


def test_stratified_const_group_pinned_rows_in_control_per_stratum() -> None:
    df = _frame(pinned_control=400)
    out = AASplitterWithStratification._inner_function(
        _dataset(df), random_state=2, grouping_fields=["s"], const_group_field="c"
    )
    split = _split(out).sort_index()
    assert (split.iloc[:400] == "control").all()
    assert (split == "control").mean() == pytest.approx(0.5, abs=0.05)


@pytest.mark.spark
def test_spark_pinned_rows_in_control(spark_session) -> None:
    df = _frame(pinned_control=150)
    out = AASplitter._inner_function(
        _dataset(df, BackendsEnum.spark, spark_session),
        random_state=3,
        const_group_field="c",
    )
    split = _split(out).sort_index()
    assert len(split) == N
    assert (split.iloc[:150] == "control").all()
    assert (split == "control").mean() == pytest.approx(0.5, abs=0.05)
