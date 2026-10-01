"""Tests for reporter helpers, DictReporter, DatasetReporter and deprecated wrappers."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from hypex.comparators import GroupDifference, GroupSizes, GroupTTest
from hypex.dataset import (
    Dataset,
    ExperimentData,
    GroupingRole,
    SmallDataset,
    TargetRole,
    TreatmentRole,
)
from hypex.experiments import OnRoleExperiment
from hypex.reporters import (
    REPORTABLE_METRICS,
    AADatasetReporter,
    ABDatasetReporter,
    ABDictReporter,
    ABTestReporter,
    DatasetReporter,
    DictReporter,
    HomoDatasetReporter,
    HomoDictReporter,
    MatchingDatasetReporter,
    MatchingDictReporter,
    MatchingQualityDatasetReporter,
    MatchingQualityDictReporter,
    OneAADictReporter,
    Reporter,
    ResultKey,
    TestDictReporter,
)
from hypex.reporters.abstract import (
    _get_index_values,
    _normalize_group_name,
    _normalize_value,
    extract_analyzer_data,
    extract_group_difference,
    extract_group_sizes,
    extract_tests,
)
from hypex.utils import ID_SPLIT_SYMBOL as S
from hypex.utils import BackendsEnum
from hypex.utils.errors import AbstractMethodError

# ---------------------------------------------------------------------------
# ResultKey
# ---------------------------------------------------------------------------
def test_result_key_parses_three_parts() -> None:
    key = ResultKey.from_id(f"GroupTTest{S}hash{S}revenue")
    assert (key.executor, key.params_hash, key.field) == ("GroupTTest", "hash", "revenue")


@pytest.mark.parametrize("raw", ["plain", f"a{S}b", f"a{S}b{S}c{S}d", ""])
def test_result_key_falls_back_for_other_shapes(raw) -> None:
    key = ResultKey.from_id(raw)
    assert key.executor == raw and key.field == raw and key.params_hash == ""


def test_result_key_is_frozen() -> None:
    key = ResultKey("a", "b", "c")
    with pytest.raises(Exception):
        key.executor = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _normalize_value
# ---------------------------------------------------------------------------
_NATIVE_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="Issue: np.float64 is a float subclass, so _normalize_value returns it unchanged "
    "instead of converting to a native float",
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (1, 1),
        (2.5, 2.5),
        ("s", "s"),
        (True, True),
        (float("nan"), None),
        pytest.param(np.float64(1.5), 1.5, marks=_NATIVE_XFAIL),
        (np.int64(3), 3),
        (np.bool_(True), True),
        (np.float64("nan"), None),
        ([4.0, 5.0], 4.0),
        ((7, 8), 7),
        pytest.param(np.array([9.0, 1.0]), 9.0, marks=_NATIVE_XFAIL),
        ([[1.0], [2.0]], 1.0),
    ],
)
def test_normalize_value(value, expected) -> None:
    result = _normalize_value(value)
    assert result == expected
    if expected is not None:
        assert type(result) in (int, float, str, bool)


def test_normalize_value_returns_unknown_objects_as_is() -> None:
    marker = object()
    assert _normalize_value(marker) is marker


def test_normalize_value_empty_list_is_unchanged() -> None:
    assert _normalize_value([]) == []


# ---------------------------------------------------------------------------
# _normalize_group_name / _get_index_values
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [("(1,)", "1"), ("(2,)", "2"), ("('a',)", "'a'"), ("1", "1"), ("(1, 2)", "(1, 2)"), (" (3,) ", "3")],
)
def test_normalize_group_name(raw, expected) -> None:
    assert _normalize_group_name(raw) == expected


def test_get_index_values_for_pandas_dataset() -> None:
    ds = SmallDataset.from_dict([{"a": 1}, {"a": 2}], roles={})
    assert [int(v) for v in _get_index_values(ds)] == [0, 1]


# ---------------------------------------------------------------------------
# Fixtures: executed comparators
# ---------------------------------------------------------------------------
@pytest.fixture
def executed() -> ExperimentData:
    rng = np.random.RandomState(0)
    n = 60
    df = pd.DataFrame(
        {
            "g": ["a"] * n + ["b"] * n,
            "y1": np.r_[rng.normal(0, 1, n), rng.normal(0.5, 1, n)],
            "y2": np.r_[rng.normal(0, 1, n), rng.normal(0, 1, n)],
        }
    )
    ds = Dataset(
        roles={"g": TreatmentRole(), "y1": TargetRole(), "y2": TargetRole()},
        data=df,
        backend=BackendsEnum.pandas,
    )
    experiment = OnRoleExperiment(
        [
            GroupSizes(grouping_role=TreatmentRole()),
            GroupDifference(grouping_role=TreatmentRole()),
            GroupTTest(compare_by="groups", grouping_role=TreatmentRole()),
        ],
        role=TargetRole(),
    )
    return experiment.execute(ExperimentData(ds))


# ---------------------------------------------------------------------------
# extract_* helpers
# ---------------------------------------------------------------------------
def test_extract_tests_returns_only_pass_and_pvalue(executed) -> None:
    result = extract_tests(executed, [GroupTTest], front=False)
    assert result
    assert all(("pass" in k) or ("p-value" in k) for k in result)
    assert {k.split(S)[0] for k in result} == {"y1", "y2"}


def test_extract_tests_key_format_front_vs_back(executed) -> None:
    back = extract_tests(executed, [GroupTTest], front=False)
    front = extract_tests(executed, [GroupTTest], front=True)
    assert f"y1{S}GroupTTest{S}p-value{S}b" in back
    assert "y1 GroupTTest p-value b" in front
    assert list(back.values()) == list(front.values())


def test_extract_tests_values_are_plain_python(executed) -> None:
    for value in extract_tests(executed, [GroupTTest], front=False).values():
        assert type(value) in (int, float, bool, str, type(None))


def test_extract_tests_for_missing_class_is_empty(executed) -> None:
    from hypex.comparators import GroupKSTest

    assert extract_tests(executed, [GroupKSTest], front=False) == {}


def test_extract_group_difference_contains_result_metrics(executed) -> None:
    result = extract_group_difference(executed, front=True)
    for metric in ("difference", "difference %", "control mean", "test mean", "ci lower", "ci upper"):
        assert f"y1 GroupDifference {metric} b" in result
        assert f"y2 GroupDifference {metric} b" in result


@pytest.mark.xfail(
    strict=True,
    reason="Issue: extract_group_difference also returns the intermediate '┆stats' tables "
    "of StatsComparator, polluting the report with keys like 'y1┆stats GroupDifference mean┆y1 a'",
)
def test_extract_group_difference_excludes_stats_tables(executed) -> None:
    result = extract_group_difference(executed, front=True)
    assert not any("stats" in k for k in result)


@pytest.mark.xfail(
    strict=True,
    reason="Issue: extract_group_sizes uses get_one_id, which returns the intermediate "
    "'┆stats' table instead of the result table, so size/percentage keys are missing",
)
def test_extract_group_sizes(executed) -> None:
    result = extract_group_sizes(executed, front=False)
    assert result[f"g{S}GroupSizes{S}control size{S}b"] == 60
    assert result[f"g{S}GroupSizes{S}test size %{S}b"] == 50.0


def test_extract_analyzer_data_empty_table_returns_empty(executed) -> None:
    class _FakeAnalyzer:
        pass

    from hypex.utils import ExperimentDataEnum

    executed.set_value(
        ExperimentDataEnum.analysis_tables,
        f"_FakeAnalyzer{S}{S}",
        SmallDataset.create_empty(),
    )
    assert extract_analyzer_data(executed, "_FakeAnalyzer") == {}


# ---------------------------------------------------------------------------
# DictReporter / DatasetReporter
# ---------------------------------------------------------------------------
def test_reporter_is_abstract() -> None:
    with pytest.raises(TypeError):
        Reporter()  # type: ignore[abstract]


def test_reporter_base_report_raises_abstract_method_error() -> None:
    class _R(Reporter):
        def report(self, data):
            return super().report(data)

    with pytest.raises(AbstractMethodError):
        _R().report(None)


def test_dict_reporter_default_is_empty(executed) -> None:
    assert DictReporter().report(executed) == {}


def test_dict_reporter_front_flag() -> None:
    assert DictReporter().front is True
    assert DictReporter(front=False).front is False


class _FixedDictReporter(DictReporter):
    def __init__(self, payload, front=True):
        super().__init__(front)
        self.payload = payload
        self.fronts_seen = []

    def _report(self, data):
        self.fronts_seen.append(self.front)
        return dict(self.payload)


def test_dataset_reporter_dict_format_returns_raw_dict(executed) -> None:
    payload = {f"y1{S}GroupTTest{S}p-value{S}b": 0.01}
    reporter = DatasetReporter(_FixedDictReporter(payload), output_format="dict")
    assert reporter.report(executed) == payload


def test_dataset_reporter_forces_back_format_and_restores_front(executed) -> None:
    inner = _FixedDictReporter({}, front=True)
    DatasetReporter(inner, output_format="dict").report(executed)
    assert inner.fronts_seen == [False]
    assert inner.front is True


def test_dataset_reporter_front_property_proxies() -> None:
    inner = DictReporter(front=True)
    reporter = DatasetReporter(inner)
    reporter.front = False
    assert inner.front is False and reporter.front is False


def test_dataset_reporter_single_row(executed) -> None:
    payload = {
        f"y1{S}GroupTTest{S}p-value{S}b": 0.01,
        f"y1{S}StatsTTest{S}pass{S}b": np.float64(1.0),
    }
    reporter = DatasetReporter(_FixedDictReporter(payload), single_row=True)
    result = reporter.report(executed)
    frame = result.backend_data.data
    assert list(frame.columns) == ["y1 TTest p-value b", "y1 TTest pass b"]
    assert len(frame) == 1


def test_dataset_reporter_single_row_empty_payload(executed) -> None:
    result = DatasetReporter(_FixedDictReporter({}), single_row=True).report(executed)
    assert len(result.columns) == 0


def test_dataset_reporter_table_format(executed) -> None:
    payload = {
        f"y1{S}GroupTTest{S}p-value{S}b": 0.01,
        f"y1{S}GroupTTest{S}pass{S}b": True,
        f"y1{S}GroupDifference{S}difference{S}b": 0.5,
        f"y1{S}GroupDifference{S}control mean{S}b": 1.0,
    }
    result = DatasetReporter(_FixedDictReporter(payload)).report(executed)
    row = result.backend_data.data.iloc[0]
    assert row["feature"] == "y1" and row["group"] == "b"
    assert row["TTest p-value"] == 0.01
    assert row["TTest pass"] == "NOT OK"
    assert row["difference"] == 0.5 and row["control mean"] == 1.0


# ---------------------------------------------------------------------------
# TestDictReporter static helpers
# ---------------------------------------------------------------------------
def test_struct_dict_nests_by_feature_group_executor_metric() -> None:
    flat = {
        f"y{S}GroupTTest{S}p-value{S}b": 0.1,
        f"y{S}GroupTTest{S}pass{S}b": False,
        f"y{S}GroupTTest{S}unknown metric{S}b": 5,  # not reportable -> dropped
        "no separator": 1,                           # skipped
        f"y{S}short{S}key": 3,                       # fewer than 4 parts -> skipped
    }
    tree = TestDictReporter._get_struct_dict(flat)
    assert tree == {"y": {"b": {"GroupTTest": {"p-value": 0.1, "pass": False}}}}


def test_reportable_metrics_contains_expected_names() -> None:
    assert {"pass", "p-value", "difference", "ci lower", "ci upper"} <= REPORTABLE_METRICS


@pytest.mark.parametrize(
    "value,label",
    [(True, "NOT OK"), (False, "OK"), ("True", "NOT OK"), ("False", "OK"), ("true", "NOT OK"),
     (None, "OK"), (1.0, "NOT OK"), (0.0, "OK")],
)
def test_pass_flag_is_converted_to_ok_labels(value, label) -> None:
    struct = {"y": {"b": {"GroupTTest": {"pass": value, "p-value": 0.5}}}}
    dataset = TestDictReporter._convert_struct_dict_to_dataset(struct)
    assert dataset.backend_data.data["TTest pass"].iloc[0] == label


def test_convert_empty_struct_gives_empty_dataset_with_columns() -> None:
    dataset = TestDictReporter._convert_struct_dict_to_dataset({})
    assert list(dataset.columns) == ["feature", "group"]
    assert len(dataset) == 0


def test_convert_struct_normalizes_test_names_and_orders_group_difference() -> None:
    struct = {
        "y": {
            "b": {
                "GroupDifference": {"difference": 1.0, "difference %": 10.0},
                "StatsTTest": {"pass": False, "p-value": 0.2},
                "GroupKSTest": {"pass": True, "p-value": 0.01},
            }
        }
    }
    frame = TestDictReporter._convert_struct_dict_to_dataset(struct).backend_data.data
    assert {"TTest pass", "TTest p-value", "KSTest pass", "KSTest p-value"} <= set(frame.columns)
    assert frame["difference %"].iloc[0] == 10.0


def test_convert_to_dataset_pipeline() -> None:
    flat = {f"y{S}GroupTTest{S}p-value{S}b": 0.3, f"y{S}GroupTTest{S}pass{S}b": False}
    frame = DatasetReporter.convert_to_dataset(flat).backend_data.data
    assert frame["TTest pass"].iloc[0] == "OK"


# ---------------------------------------------------------------------------
# ABTestReporter on executed data
# ---------------------------------------------------------------------------
def test_ab_reporter_requires_analyzer_table(executed) -> None:
    with pytest.raises(Exception):
        ABTestReporter(DictReporter(), output_format="dict").report(executed)


def test_report_variance_reductions_message_when_missing(executed) -> None:
    assert "No variance reduction data" in ABTestReporter.report_variance_reductions(executed)


# ---------------------------------------------------------------------------
# Deprecated wrappers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "factory",
    [
        lambda: ABDictReporter(),
        lambda: ABDatasetReporter(),
        lambda: HomoDictReporter(),
        lambda: HomoDatasetReporter(),
        lambda: OneAADictReporter(),
        lambda: AADatasetReporter(),
        lambda: MatchingDictReporter(),
        lambda: MatchingDatasetReporter(),
        lambda: MatchingQualityDictReporter(),
        lambda: MatchingQualityDatasetReporter(),
    ],
    ids=[
        "ABDict", "ABDataset", "HomoDict", "HomoDataset", "OneAADict", "AADataset",
        "MatchingDict", "MatchingDataset", "MatchingQualityDict", "MatchingQualityDataset",
    ],
)
def test_deprecated_wrappers_warn(factory) -> None:
    with pytest.warns(DeprecationWarning):
        reporter = factory()
    assert isinstance(reporter, DatasetReporter)


def test_non_deprecated_reporter_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ABTestReporter(DictReporter())


def test_deprecated_dict_wrapper_outputs_dict_format() -> None:
    with pytest.warns(DeprecationWarning):
        reporter = ABDictReporter(front=False)
    assert reporter.output_format == "dict"
    assert reporter.front is False
    with pytest.warns(DeprecationWarning):
        assert ABDatasetReporter().output_format == "dataset"


def test_star_import_exposes_public_api() -> None:
    namespace: dict = {}
    exec("from hypex.reporters import *", namespace)
    import hypex.reporters as reporters

    assert all(name in namespace for name in reporters.__all__)
