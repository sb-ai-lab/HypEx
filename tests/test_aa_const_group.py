"""ConstGroupRole behaviour of AASplitter on the pandas and Spark backends."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from hypex.dataset import (
    Dataset,
    ExperimentData,
    InfoRole,
    StratificationRole,
    TargetRole,
)
from hypex.dataset.roles import ConstGroupRole
from hypex.splitters.aa import AASplitter, AASplitterWithStratification
from hypex.utils import BackendsEnum

BACKENDS = ["pandas", "spark"]
CONST_COLUMN = "const_group"
DEFAULT_LABEL_MAP = {0: "control", 1: "test_1"}
UNKNOWN_LABEL_MESSAGE = (
    "Unknown constant group {label!r} in column 'const_group'. Expected one of "
    "['control', 'test', 'test_1'], or a missing value (None / np.nan / 'nan') "
    "for a row that takes part in the split."
)


def _frame(const_values, index=None):
    n = len(const_values)
    return pd.DataFrame(
        {
            "id": list(range(n)),
            "target": [float(i) for i in range(n)],
            "strat": [str(i % 2) for i in range(n)],
            CONST_COLUMN: list(const_values),
        },
        index=index,
    )


def _roles(with_const=True):
    roles = {
        "id": InfoRole(),
        "target": TargetRole(),
        "strat": StratificationRole(),
    }
    if with_const:
        roles[CONST_COLUMN] = ConstGroupRole()
    return roles


def _pinned_values():
    """60 pinned 'control', 40 pinned 'test', 400 free (None)."""
    return ["control"] * 60 + ["test"] * 40 + [None] * 400


@pytest.fixture
def make_dataset(request):
    """Factory: (pd.DataFrame, roles, backend) -> Dataset."""

    def _make(pdf, roles, backend):
        dataset = Dataset(roles=deepcopy(roles), data=pdf.copy())
        if backend == "spark":
            session = request.getfixturevalue("spark_session")
            dataset = dataset.to_backend(BackendsEnum.spark, session=session)
        return dataset

    return _make


def _as_pandas(dataset):
    raw = dataset.data
    return raw.to_pandas() if hasattr(raw, "to_pandas") else raw


def _split_frame(dataset):
    return _as_pandas(dataset).sort_index()


def _split(dataset, **kwargs):
    kwargs.setdefault("random_state", 42)
    kwargs.setdefault("control_size", 0.5)
    kwargs.setdefault("const_group_field", CONST_COLUMN)
    return AASplitter._inner_function(dataset, **kwargs)


@pytest.mark.parametrize("backend", BACKENDS)
def test_const_group_plan_rescales_control_size(make_dataset, backend):
    dataset = make_dataset(_frame(_pinned_values()), _roles(), backend)

    translation, free_size, control_size = AASplitter._const_group_plan(
        dataset, CONST_COLUMN, DEFAULT_LABEL_MAP, 0.5
    )

    assert translation == {"control": "control", "test": "test_1"}
    assert free_size == 400
    assert control_size == pytest.approx(0.475)


@pytest.mark.parametrize("backend", BACKENDS)
def test_inner_function_pins_and_splits(make_dataset, backend):
    dataset = make_dataset(_frame(_pinned_values()), _roles(), backend)

    result = _split(dataset)
    frame = _split_frame(result)

    assert list(result.columns) == ["split"]
    assert len(frame) == 500
    assert frame.index.nunique() == 500
    assert set(frame.loc[0:59, "split"]) == {"control"}
    assert set(frame.loc[60:99, "split"]) == {"test_1"}
    assert set(frame.loc[100:499, "split"]) <= {"control", "test_1"}
    free_share = (frame.loc[100:499, "split"] == "control").mean()
    assert abs(free_share - 0.475) <= 0.10, free_share
    total_share = (frame["split"] == "control").mean()
    assert abs(total_share - 0.5) <= 0.10, total_share


@pytest.mark.parametrize("backend", BACKENDS)
def test_inner_function_missing_labels_join_the_split(make_dataset, backend):
    free = (["nan"] * 100) + (["None"] * 100) + ([" NaN "] * 100) + ([""] * 100)
    dataset = make_dataset(
        _frame(["control"] * 60 + ["test"] * 40 + free), _roles(), backend
    )

    frame = _split_frame(_split(dataset))

    assert len(frame) == 500
    assert frame.index.nunique() == 500
    assert set(frame.loc[0:59, "split"]) == {"control"}
    assert set(frame.loc[60:99, "split"]) == {"test_1"}
    assert set(frame.loc[100:499, "split"]) <= {"control", "test_1"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_inner_function_no_pinned_rows(make_dataset, backend):
    dataset = make_dataset(_frame(["nan"] * 500), _roles(), backend)

    frame = _split_frame(_split(dataset))

    assert len(frame) == 500
    assert frame.index.nunique() == 500
    assert set(frame["split"]) == {"control", "test_1"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_inner_function_all_rows_pinned(make_dataset, backend):
    dataset = make_dataset(
        _frame(["control"] * 250 + ["test"] * 250), _roles(), backend
    )

    frame = _split_frame(_split(dataset))

    assert len(frame) == 500
    assert frame.index.nunique() == 500
    assert frame["split"].value_counts().to_dict() == {"control": 250, "test_1": 250}


@pytest.mark.parametrize("backend", BACKENDS)
def test_inner_function_only_test_pinned(make_dataset, backend):
    dataset = make_dataset(_frame(["test"] * 100 + [None] * 400), _roles(), backend)

    translation, free_size, control_size = AASplitter._const_group_plan(
        dataset, CONST_COLUMN, DEFAULT_LABEL_MAP, 0.5
    )
    assert translation == {"test": "test_1"}
    assert free_size == 400
    assert control_size == pytest.approx(0.625)

    frame = _split_frame(_split(dataset))
    assert len(frame) == 500
    assert frame.index.nunique() == 500
    assert set(frame.loc[0:99, "split"]) == {"test_1"}
    free_share = (frame.loc[100:499, "split"] == "control").mean()
    assert abs(free_share - 0.625) <= 0.10, free_share


@pytest.mark.parametrize(
    "bad_label", ["bogus", "Control", " control", "TEST", "test_2"]
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_inner_function_rejects_bad_const_labels(make_dataset, backend, bad_label):
    values = [bad_label] * 10 + ["control"] * 50 + [None] * 440
    dataset = make_dataset(_frame(values), _roles(), backend)

    with pytest.raises(ValueError) as excinfo:
        _split(dataset)

    assert str(excinfo.value) == UNKNOWN_LABEL_MESSAGE.format(label=bad_label)


@pytest.mark.parametrize("backend", BACKENDS)
def test_inner_function_rejects_sentinel_const_value(make_dataset, backend):
    from hypex.splitters.aa import _FREE_CONST_SENTINEL

    values = [_FREE_CONST_SENTINEL] * 10 + [None] * 490
    dataset = make_dataset(_frame(values), _roles(), backend)

    with pytest.raises(ValueError) as excinfo:
        _split(dataset)

    assert str(excinfo.value) == UNKNOWN_LABEL_MESSAGE.format(
        label=_FREE_CONST_SENTINEL
    )


def test_inner_function_categorical_const_column(make_dataset):
    """A pandas `category` dtype const column must not crash fillna() with

    ``TypeError: Cannot setitem on a Categorical with a new category``
    (regression: review round 1, R1-01). dev/master returns a correct split
    for the identical input, so this is a real regression, not an
    unsupported case. Spark has no categorical dtype reachable on this
    path, so this test is pandas-only.
    """
    pdf = _frame(["control"] * 8 + [None] * 12)
    pdf[CONST_COLUMN] = pdf[CONST_COLUMN].astype("category")
    dataset = make_dataset(pdf, _roles(), "pandas")

    frame = _split_frame(_split(dataset))

    assert len(frame) == 20
    assert frame.index.nunique() == 20
    assert set(frame.loc[0:7, "split"]) == {"control"}
    assert set(frame["split"]) <= {"control", "test_1"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_inner_function_const_column_named_count(make_dataset, backend):
    """A const column literally named 'count' must not collide with

    ``value_counts()``'s own 'count' output column (regression: review
    round 1, R1-02). dev/master handles the same input via ``groupby``.
    """
    pdf = _frame(["control"] * 5 + [None] * 5).rename(columns={CONST_COLUMN: "count"})
    roles = _roles(with_const=False)
    roles["count"] = ConstGroupRole()
    dataset = make_dataset(pdf, roles, backend)

    frame = _split_frame(
        AASplitter._inner_function(
            dataset, random_state=42, control_size=0.5, const_group_field="count"
        )
    )

    assert len(frame) == 10
    assert frame.index.nunique() == 10
    assert set(frame.loc[0:4, "split"]) == {"control"}
    assert set(frame["split"]) <= {"control", "test_1"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_inner_function_three_groups_with_pinned(make_dataset, backend):
    values = (
        ["control"] * 30 + ["test_1"] * 30 + ["test_2"] * 30 + [None] * 410
    )
    dataset = make_dataset(_frame(values), _roles(), backend)

    frame = _split_frame(_split(dataset, groups_sizes=[0.34, 0.33, 0.33]))

    assert len(frame) == 500
    assert frame.index.nunique() == 500
    assert set(frame["split"]) == {"control", "test_1", "test_2"}
    assert set(frame.loc[0:29, "split"]) == {"control"}
    assert set(frame.loc[30:59, "split"]) == {"test_1"}
    assert set(frame.loc[60:89, "split"]) == {"test_2"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_inner_function_deterministic_and_order_independent(make_dataset, backend):
    pdf = _frame(_pinned_values())
    first = _split_frame(_split(make_dataset(pdf, _roles(), backend)))
    second = _split_frame(_split(make_dataset(pdf, _roles(), backend)))
    reversed_rows = _split_frame(
        _split(make_dataset(pdf.iloc[::-1], _roles(), backend))
    )

    assert first["split"].to_dict() == second["split"].to_dict()
    assert first["split"].to_dict() == reversed_rows["split"].to_dict()


@pytest.mark.parametrize("backend", BACKENDS)
def test_inner_function_without_const_column(make_dataset, backend):
    pdf = _frame(_pinned_values()).drop(columns=[CONST_COLUMN])
    dataset = make_dataset(pdf, _roles(with_const=False), backend)

    frame = _split_frame(_split(dataset, const_group_field=None))

    assert len(frame) == 500
    assert frame.index.nunique() == 500
    assert set(frame["split"]) == {"control", "test_1"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_inner_function_two_rows(make_dataset, backend):
    dataset = make_dataset(_frame(["control", None]), _roles(), backend)

    frame = _split_frame(_split(dataset))

    assert len(frame) == 2
    assert frame.loc[0, "split"] == "control"
    assert frame.loc[1, "split"] in {"control", "test_1"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_inner_function_duplicate_index(make_dataset, backend):
    dataset = make_dataset(
        _frame(["control"] * 4 + [None] * 6, index=[0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
        _roles(),
        backend,
    )

    frame = _split_frame(_split(dataset))

    assert len(frame) == 10
    assert (frame["split"] == "control").sum() >= 4
    assert set(frame["split"]) <= {"control", "test_1"}


def test_inner_function_empty_dataset(make_dataset):
    dataset = make_dataset(_frame([]), _roles(), "pandas")

    result = _split(dataset)

    assert len(result) == 0
    assert list(result.columns) == ["split"]


def test_inner_function_multiindex(make_dataset):
    index = pd.MultiIndex.from_tuples(
        [(i // 5, i % 5) for i in range(20)], names=["a", "b"]
    )
    dataset = make_dataset(
        _frame(["control"] * 4 + [None] * 16, index=index), _roles(), "pandas"
    )

    frame = _split_frame(_split(dataset))

    assert len(frame) == 20
    assert list(frame.index.names) == ["a", "b"]
    assert set(frame["split"]) <= {"control", "test_1"}
    assert set(frame.loc[(0, slice(None)), "split"]) == {"control"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_execute_with_const_group(make_dataset, backend):
    dataset = make_dataset(_frame(_pinned_values()), _roles(), backend)
    splitter = AASplitter(control_size=0.5, random_state=42, save_groups=True)

    result = splitter.execute(ExperimentData(dataset))

    column = splitter._id
    frame = _as_pandas(result.ds).sort_index()
    assert column in frame.columns
    assert frame[column].notna().sum() == 500
    assert set(frame[column]) <= {"control", "test_1"}
    assert set(frame.loc[0:59, column]) == {"control"}
    assert set(frame.loc[60:99, column]) == {"test_1"}
    assert set(result.groups[column]) == {"control", "test_1"}
    assert sum(len(g) for g in result.groups[column].values()) == 500


@pytest.mark.parametrize("backend", BACKENDS)
def test_stratified_execute_pins_per_stratum(make_dataset, backend):
    dataset = make_dataset(_frame(_pinned_values()), _roles(), backend)
    splitter = AASplitterWithStratification(
        control_size=0.5, random_state=42, save_groups=False
    )

    result = splitter.execute(ExperimentData(dataset))

    column = splitter._id
    frame = _as_pandas(result.ds).sort_index()
    assert len(frame) == 500
    assert frame[column].notna().sum() == 500
    assert set(frame.loc[0:59, column]) == {"control"}
    assert set(frame.loc[60:99, column]) == {"test_1"}
    for stratum in ("0", "1"):
        subset = frame[frame["strat"] == stratum]
        assert len(subset) == 250
        share = (subset[column] == "control").mean()
        assert abs(share - 0.5) <= 0.12, (stratum, share)
