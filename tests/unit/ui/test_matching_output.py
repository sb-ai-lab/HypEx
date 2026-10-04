"""Tests for MatchingOutput built on hand-assembled ExperimentData."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from hypex.analyzers.matching import MatchingAnalyzer
from hypex.dataset import (
    AdditionalMatchingRole,
    Dataset,
    ExperimentData,
    FeatureRole,
    GroupingRole,
    TargetRole,
    TreatmentRole,
)
from hypex.operators import MatchingMetrics
from hypex.ui.matching import MatchingOutput
from hypex.utils import ID_SPLIT_SYMBOL as S
from hypex.utils import MATCHING_INDEXES_SPLITTER_SYMBOL as M
from hypex.utils import BackendsEnum, ExperimentDataEnum

N = 30


def _make_output(**kwargs) -> MatchingOutput:
    """Build the output while ignoring the deprecated-reporter warning it emits."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return MatchingOutput(**kwargs)


@pytest.fixture
def matched_data():
    rng = np.random.RandomState(0)
    x = np.r_[rng.uniform(0, 10, N), rng.uniform(2, 12, N)]
    treat = np.r_[np.zeros(N, dtype=int), np.ones(N, dtype=int)]
    y = 2.0 * x + 5.0 * treat + rng.normal(0, 1, 2 * N)
    nn = np.empty(2 * N, dtype=int)
    for i in range(2 * N):
        pool = np.where(treat != treat[i])[0]
        nn[i] = pool[np.argmin(np.abs(x[pool] - x[i]))]
    df = pd.DataFrame({"treat": treat, "x": x, "y": y, "nn": nn})
    ds = Dataset(
        roles={
            "treat": TreatmentRole(),
            "x": FeatureRole(),
            "y": TargetRole(),
            "nn": AdditionalMatchingRole(),
        },
        data=df,
        backend=BackendsEnum.pandas,
    )
    data = MatchingMetrics(grouping_role=TreatmentRole()).execute(ExperimentData(ds))
    data = MatchingAnalyzer().execute(data)
    data.set_value(
        ExperimentDataEnum.additional_fields,
        f"FaissNearestNeighbors{S}h{S}{S}0",
        list(nn),
        role=AdditionalMatchingRole(),
    )
    return data, nn


# --------------------------------------------------------------------------
# _reformat_summary
# --------------------------------------------------------------------------
def test_reformat_summary_groups_by_metric() -> None:
    summary = {f"Effect Size{S}ATT": 1.0, f"Effect Size{S}ATC": 2.0, "plain": 9}
    assert MatchingOutput._reformat_summary(summary) == {
        "Effect Size": {"ATT": 1.0, "ATC": 2.0}
    }


def test_reformat_summary_three_part_key_prefixes_third_part() -> None:
    assert MatchingOutput._reformat_summary({f"p{S}row{S}grp": 3}) == {
        "grp p": {"row": 3}
    }


def test_reformat_summary_indexes_flat_and_nested() -> None:
    summary = {f"indexes{S}0": "1", f"indexes{S}g{S}1": "2"}
    assert MatchingOutput._reformat_summary(summary) == {
        "indexes": {"0": "1", "g": {"1": "2"}}
    }


# --------------------------------------------------------------------------
# extract on a pandas ExperimentData
# --------------------------------------------------------------------------
def test_extract_summary_matches_analyzer_table_rounded(matched_data) -> None:
    data, _ = matched_data
    expected = data.analysis_tables[next(iter(data.analysis_tables))].raw_data.round(2)
    output = _make_output()
    output.extract(data)
    pd.testing.assert_frame_equal(output.summary.raw_data, expected, check_dtype=False)
    assert list(output.summary.columns) == [
        "Effect Size",
        "Standard Error",
        "P-value",
        "CI Lower",
        "CI Upper",
    ]


def test_extract_skips_indexes_and_full_data_by_default(matched_data) -> None:
    data, _ = matched_data
    output = _make_output()
    output.extract(data)
    assert output.full_data.is_empty()
    assert output.indexes.is_empty()


def test_extract_compute_indexes_returns_matched_indexes(matched_data) -> None:
    data, nn = matched_data
    output = _make_output(compute_indexes=True, extract_full_data=True)
    output.extract(data)
    assert output.indexes["indexes_0"].raw_data.iloc[:, 0].tolist() == nn.tolist()


def test_extract_full_data_adds_matched_columns(matched_data) -> None:
    data, nn = matched_data
    output = _make_output(compute_indexes=True, extract_full_data=True)
    output.extract(data)
    full = output.full_data.raw_data
    assert len(full) == 2 * N
    original = data.ds.raw_data
    # matched_0 columns hold the covariates of the nearest neighbour
    np.testing.assert_allclose(
        full["x_matched_0"].to_numpy(), original["x"].to_numpy()[nn]
    )
    np.testing.assert_allclose(
        full["y_matched_0"].to_numpy(), original["y"].to_numpy()[nn]
    )
    # opposite groups are matched
    assert (full["treat_matched_0"].to_numpy() != full["treat"].to_numpy()).all()


def test_extract_full_data_with_unmatched_rows(matched_data) -> None:
    data, nn = matched_data
    output = _make_output()
    unmatched = Dataset(
        roles={"indexes_0": AdditionalMatchingRole()},
        data=pd.DataFrame({"indexes_0": np.where(np.arange(2 * N) < 5, -1, nn)}),
        backend=BackendsEnum.pandas,
    )
    output._extract_full_data(data, unmatched)
    full = output.full_data.raw_data
    assert full["x_matched_0"].iloc[:5].isna().all()
    assert full["x_matched_0"].iloc[5:].notna().all()


def test_match_pandas_all_unmatched_returns_empty(matched_data) -> None:
    data, _ = matched_data
    output = _make_output()
    t_indexes = Dataset(
        roles={"i": AdditionalMatchingRole()},
        data=pd.DataFrame({"i": [-1, -1]}),
        backend=BackendsEnum.pandas,
    )
    assert output._match_pandas(data, t_indexes, "i").is_empty()


# --------------------------------------------------------------------------
# _extract_driver_indexes branches
# --------------------------------------------------------------------------
def _grouped_data() -> ExperimentData:
    df = pd.DataFrame({"g": ["a", "a", "b", "b"], "x": [1.0, 2.0, 3.0, 4.0]})
    return ExperimentData(
        Dataset(
            roles={"g": GroupingRole(), "x": FeatureRole()},
            data=df,
            backend=BackendsEnum.pandas,
        )
    )


def test_driver_indexes_single_matching_length() -> None:
    output = _make_output()
    output.summary = {"indexes": M.join(["2", "3", "0", "1"])}
    indexes = output._extract_driver_indexes(_grouped_data(), {})
    assert indexes["indexes"].raw_data.iloc[:, 0].tolist() == [2, 3, 0, 1]


def test_driver_indexes_single_length_mismatch_warns() -> None:
    output = _make_output()
    output.summary = {"indexes": M.join(["2", "3"])}
    with pytest.warns(UserWarning, match="Alignment skipped"):
        indexes = output._extract_driver_indexes(_grouped_data(), {})
    assert len(indexes) == 2


def test_driver_indexes_absent_gives_empty() -> None:
    output = _make_output()
    output.summary = {}
    assert output._extract_driver_indexes(_grouped_data(), {}).is_empty()


def test_driver_indexes_flat_grouped() -> None:
    output = _make_output()
    summary = {"indexes": {"0": M.join(["2", "3", "0", "1"])}}
    indexes = output._extract_driver_indexes(_grouped_data(), summary)
    assert indexes["indexes_0"].raw_data.iloc[:, 0].tolist() == [2, 3, 0, 1]
    assert "indexes" not in summary


def test_driver_indexes_flat_grouped_length_mismatch_warns() -> None:
    output = _make_output()
    with pytest.warns(UserWarning, match="for group '0'"):
        output._extract_driver_indexes(
            _grouped_data(), {"indexes": {"0": M.join(["2", "3"])}}
        )


def test_driver_indexes_nested_grouped() -> None:
    output = _make_output()
    summary = {"indexes": {"0": {"a": M.join(["2", "3"]), "b": M.join(["0", "1"])}}}
    indexes = output._extract_driver_indexes(_grouped_data(), summary)
    assert indexes["indexes_0"].raw_data.iloc[:, 0].tolist() == [2, 3, 0, 1]


def test_collect_grouped_indexes_aligns_to_group_rows() -> None:
    result = MatchingOutput._collect_grouped_indexes(
        _grouped_data(), {"a": M.join(["9", "8"]), "b": M.join(["7", "6"])}
    )
    assert result["indexes"].raw_data.iloc[:, 0].tolist() == [9, 8, 7, 6]
    assert result.raw_data.index.tolist() == [0, 1, 2, 3]


def test_matching_output_construction_does_not_use_deprecated_reporters() -> None:
    MatchingOutput()
