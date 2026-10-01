"""Tests for hypex.utils.naming."""
from __future__ import annotations

import pytest

from hypex.utils.constants import ID_SPLIT_SYMBOL as S
from hypex.utils.constants import NAME_BORDER_SYMBOL as B
from hypex.utils.constants import TEST_NAME_NORMALIZATION
from hypex.utils.naming import METRIC_SUFFIXES, _parse_metric_col, normalize_test_name


@pytest.mark.parametrize("raw,expected", sorted(TEST_NAME_NORMALIZATION.items()))
def test_normalize_known_names(raw, expected) -> None:
    assert normalize_test_name(raw) == expected


@pytest.mark.parametrize("raw", ["Unknown", "", "ttest", "StatsTTestX"])
def test_normalize_unknown_names_are_unchanged(raw) -> None:
    assert normalize_test_name(raw) == raw


def test_normalization_is_idempotent() -> None:
    for raw in TEST_NAME_NORMALIZATION:
        once = normalize_test_name(raw)
        assert normalize_test_name(once) == once


def test_stats_and_group_variants_normalize_identically() -> None:
    assert normalize_test_name("StatsTTest") == normalize_test_name("GroupTTest") == "TTest"
    assert normalize_test_name("StatsKSTest") == normalize_test_name("GroupKSTest") == "KSTest"
    assert normalize_test_name("StatsChi2Test") == normalize_test_name("GroupChi2Test") == "Chi2Test"


# ---------------------------------------------------------------------------
# _parse_metric_col
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "col,expected",
    [
        ("y GroupTTest p-value b", ("y", "GroupTTest", "p-value", "b")),
        ("y GroupTTest pass b", ("y", "GroupTTest", "pass", "b")),
        ("y GroupDifference control mean b", ("y", "GroupDifference", "control mean", "b")),
        ("y GroupDifference test mean b", ("y", "GroupDifference", "test mean", "b")),
        ("y GroupDifference difference b", ("y", "GroupDifference", "difference", "b")),
        ("y GroupDifference difference % b", ("y", "GroupDifference", "difference %", "b")),
        ("y StatsTTest p-value", ("y", "StatsTTest", "p-value", "")),
        ("TTest p-value b", ("", "TTest", "p-value", "b")),
    ],
)
def test_space_separated_columns(col, expected) -> None:
    assert _parse_metric_col(col) == expected


def test_difference_percent_is_not_confused_with_difference() -> None:
    assert _parse_metric_col("y GroupDifference difference % b")[2] == "difference %"
    assert _parse_metric_col("y GroupDifference difference b")[2] == "difference"


@pytest.mark.parametrize(
    "col,expected",
    [
        (f"y{S}GroupTTest{S}p-value{S}b", ("y", "GroupTTest", "p-value", "b")),
        (f"y{S}GroupTTest{S}p-value", ("y", "GroupTTest", "p-value", "")),
        (f"y{S}GroupTTest", ("", "", "", "")),
        (f"a{S}b{S}c{S}d{S}e", ("a", "b", "c", "d")),
    ],
)
def test_legacy_id_split_columns(col, expected) -> None:
    assert _parse_metric_col(col) == expected


@pytest.mark.parametrize("col", [f"y{B}stats GroupDifference mean{B}y a", f"x{B}y", ""])
def test_stats_and_empty_columns_are_skipped(col) -> None:
    assert _parse_metric_col(col) == ("", "", "", "")


def test_unparseable_column_returns_empty_parts() -> None:
    assert _parse_metric_col("nothing useful here") == ("", "", "", "")


def test_all_metric_suffixes_are_parseable() -> None:
    for metric in METRIC_SUFFIXES:
        feature, test, parsed_metric, group = _parse_metric_col(f"f Test {metric} g")
        assert (feature, test, parsed_metric, group) == ("f", "Test", metric, "g")


def test_group_names_with_spaces_are_joined() -> None:
    assert _parse_metric_col("y GroupTTest p-value test 1")[3] == "test 1"
