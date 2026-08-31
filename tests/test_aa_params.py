"""Regression tests for the parameters AATest passes down to the splitter."""

import numpy as np
import pandas as pd
import pytest

from hypex import AATest
from hypex.dataset import Dataset, InfoRole, TargetRole
from hypex.dataset.roles import ConstGroupRole
from hypex.splitters import AASplitter


def _make_data(n: int = 300, seed: int = 3) -> Dataset:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "user_id": np.arange(n),
            "pre_spends": rng.normal(100, 10, n),
        }
    )
    return Dataset(
        roles={"user_id": InfoRole(), "pre_spends": TargetRole()},
        data=df,
    )


def test_sample_size_together_with_groups_sizes():
    """The second pass added by sample_size used to get shifted arguments.

    ``groups_sizes`` landed in the ``additional_params`` slot of
    ``_prepare_params`` and blew up on ``params.update(list)``.
    """
    result = AATest(
        sample_size=0.3,
        groups_sizes=[0.5, 0.2, 0.3],
        random_states=range(3),
    ).execute(_make_data())

    groups = set(result.best_split["split"].get_values(column="split"))
    assert groups == {"control", "test_1", "test_2"}


def _with_const_groups(data: Dataset, values) -> Dataset:
    return data.add_column(list(values), role={"forced": ConstGroupRole(str)})


def _split_of(result) -> list:
    return list(result.best_split["split"].get_values(column="split"))


def test_all_rows_in_const_groups_are_kept_as_the_split():
    """Every row can already be pinned to a constant group - then there is
    nothing to split and the constant assignment is the split itself."""
    data = _make_data()
    rng = np.random.default_rng(7)
    forced = rng.choice(["control", "test"], len(data))
    data = _with_const_groups(data, forced)

    result = AATest(random_states=range(2)).execute(data)

    split = _split_of(result)
    assert len(split) == len(data)
    expected = ["control" if value == "control" else "test_1" for value in forced]
    assert split == expected


@pytest.mark.parametrize("label", ["tset", "test_5"], ids=["typo", "not-requested"])
def test_unknown_constant_group_is_reported(label):
    """A value that is neither ``control`` nor a group of the split used to end up
    in ``test_1`` without a word - a typo and a group that was not asked for
    alike. Labels that stand for a missing value are a separate story."""
    data = _make_data()
    values = np.where(np.arange(len(data)) % 2 == 0, "control", label)
    data = _with_const_groups(data, values)

    with pytest.raises(ValueError, match="Unknown constant group"):
        AASplitter._inner_function(data, const_group_field="forced")


def test_rows_can_be_pinned_to_a_particular_test_group():
    """``test_2`` used to be silently merged into ``test_1``."""
    data = _make_data()
    labels = ["control", "test_1", "test_2", None]
    values = [labels[i % len(labels)] for i in range(len(data))]
    data = _with_const_groups(data, values)

    split = AASplitter._inner_function(
        data, random_state=1, const_group_field="forced", groups_sizes=[0.4, 0.3, 0.3]
    )

    for pinned, assigned in zip(values, split):
        if pinned is not None:
            assert assigned == pinned


def test_test_is_an_alias_for_the_first_test_group():
    """The documented convention of a two-group split is ``control`` / ``test``."""
    data = _make_data()
    values = np.where(np.arange(len(data)) % 2 == 0, "control", "test")
    data = _with_const_groups(data, values)

    split = AASplitter._inner_function(data, random_state=1, const_group_field="forced")

    assert [
        "control" if value == "control" else "test_1" for value in values
    ] == split


def test_all_rows_in_const_groups_with_groups_sizes():
    """The same holds when the group sizes are set explicitly."""
    data = _make_data()
    rng = np.random.default_rng(7)
    forced = rng.choice(["control", "test"], len(data))
    data = _with_const_groups(data, forced)

    result = AATest(groups_sizes=[0.5, 0.2, 0.3], random_states=range(2)).execute(data)

    split = _split_of(result)
    assert len(split) == len(data)
    assert set(split) <= {"control", "test_1"}
    assert None not in split


@pytest.mark.parametrize("free_share", [0.5, 0.02])
def test_const_groups_leave_the_split_working(free_share):
    """As long as some rows are free, they are split as usual - even if almost
    every row is pinned to a constant group."""
    data = _make_data()
    rng = np.random.default_rng(11)
    forced = rng.choice(
        [None, "control", "test"],
        len(data),
        p=[free_share, (1 - free_share) / 2, (1 - free_share) / 2],
    )
    data = _with_const_groups(data, forced)

    result = AATest(random_states=range(2)).execute(data)

    groups = set(_split_of(result))
    assert "control" in groups
    assert any(group.startswith("test") for group in groups)
    assert len(result.resume.data) > 0


def test_missing_values_pandas_wrote_as_strings_stay_free():
    """Assigning a string into a column that does not exist yet makes pandas
    write the missing values of the other rows as the string 'nan' - the
    pattern of the A/A tutorial. Those rows are unpinned like any other missing
    value; they used to be forced into test_1 instead."""
    frame = pd.DataFrame(
        {
            "user_id": np.arange(300),
            "treat": [0.0, 1.0, np.nan] * 100,
            "pre_spends": np.random.default_rng(3).normal(100, 10, 300),
        }
    )
    frame.loc[frame["treat"] == 0, "forced"] = "control"
    frame.loc[frame["treat"] == 1, "forced"] = "test"
    assert (frame["forced"] == "nan").sum() == 100, "pandas no longer writes 'nan'"

    data = Dataset(
        roles={
            "user_id": InfoRole(),
            "pre_spends": TargetRole(),
            "forced": ConstGroupRole(str),
        },
        data=frame,
    )

    split = pd.Series(
        AASplitter._inner_function(data, random_state=1, const_group_field="forced")
    )

    assert set(split[frame["forced"] == "control"]) == {"control"}
    assert set(split[frame["forced"] == "test"]) == {"test_1"}
    # the rows pandas marked as 'nan' were split, not pinned
    assert set(split[frame["forced"] == "nan"]) == {"control", "test_1"}
