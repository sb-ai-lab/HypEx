"""Tests for IfExecutor and IfAAExecutor branching logic."""

from __future__ import annotations

import pandas as pd
import pytest

from hypex.dataset import Dataset, ExperimentData, FeatureRole
from hypex.executor.executor import Executor, IfExecutor
from hypex.forks.aa import IfAAExecutor
from hypex.utils import BackendsEnum


class _Marker(Executor):
    """Executor that records that it ran."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(key=name)

    def execute(self, data):
        data.calls = [*getattr(data, "calls", []), self.name]
        return data


class _Rule(IfExecutor):
    def __init__(self, outcome: bool, **kwargs):
        self.outcome = outcome
        super().__init__(**kwargs)

    def check_rule(self, data, **kwargs) -> bool:
        return self.outcome


@pytest.fixture
def exp_data() -> ExperimentData:
    ds = Dataset(
        roles={"x": FeatureRole()},
        data=pd.DataFrame({"x": [1.0, 2.0]}),
        backend=BackendsEnum.pandas,
    )
    return ExperimentData(ds)


def test_true_without_executors_sets_response_true(exp_data) -> None:
    rule = _Rule(True)
    out = rule.execute(exp_data)
    assert out.variables[rule.id]["response"] is True


def test_false_without_executors_sets_response_false(exp_data) -> None:
    rule = _Rule(False)
    out = rule.execute(exp_data)
    assert out.variables[rule.id]["response"] is False


def test_true_runs_only_if_executor(exp_data) -> None:
    rule = _Rule(True, if_executor=_Marker("if"), else_executor=_Marker("else"))
    out = rule.execute(exp_data)
    assert out.calls == ["if"]
    assert rule.id not in out.variables


def test_false_runs_only_else_executor(exp_data) -> None:
    rule = _Rule(False, if_executor=_Marker("if"), else_executor=_Marker("else"))
    out = rule.execute(exp_data)
    assert out.calls == ["else"]


def test_true_with_only_else_executor_sets_response(exp_data) -> None:
    rule = _Rule(True, else_executor=_Marker("else"))
    out = rule.execute(exp_data)
    assert out.variables[rule.id]["response"] is True
    assert not hasattr(out, "calls")


def test_if_executor_is_abstract() -> None:
    with pytest.raises(TypeError):
        IfExecutor()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# IfAAExecutor
# ---------------------------------------------------------------------------
class _FakeTable:
    def __init__(self, values: dict):
        self._values = values
        self.columns = list(values)

    def select(self, column):
        value = self._values[column]

        class _Sel:
            @staticmethod
            def iget_values(row, col):
                return value

        return _Sel()


class _FakeData:
    def __init__(self, values: dict):
        self.analysis_tables = {"tid": _FakeTable(values)}

    def get_one_id(self, *_args, **_kwargs):
        return "tid"


@pytest.mark.parametrize(
    "values,expected",
    [
        ({"x pass": 0, "y pass": 0}, True),
        ({"x pass": 1, "y pass": 0}, False),
        ({"x pass": 0, "y pass": 2}, False),
    ],
)
def test_all_features_passed_rule(values, expected) -> None:
    ex = IfAAExecutor(all_features_passed=True)
    assert ex.check_rule(_FakeData(values)) is expected


def test_all_features_passed_counts_only_pass_columns() -> None:
    ex = IfAAExecutor(all_features_passed=True)
    data = _FakeData({"x pass": 0, "x mean p-value": 5})
    assert ex.check_rule(data) is True


@pytest.mark.parametrize(
    "values,expected",
    [({"x pass": 0}, False), ({"x pass": 1}, True), ({"x pass": 3}, True)],
)
def test_sample_size_rule(values, expected) -> None:
    ex = IfAAExecutor(sample_size=0.5)
    assert ex.check_rule(_FakeData(values)) is expected


def test_no_rule_configured_returns_false() -> None:
    assert IfAAExecutor().check_rule(_FakeData({"x pass": 0})) is False


def test_all_features_passed_takes_priority_over_sample_size() -> None:
    ex = IfAAExecutor(sample_size=0.5, all_features_passed=True)
    assert ex.check_rule(_FakeData({"x pass": 0})) is True
