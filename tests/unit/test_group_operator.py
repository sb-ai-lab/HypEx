"""Tests for GroupOperator (the abstract base of SMD / MatchingMetrics / Bias)."""

from __future__ import annotations

import pandas as pd
import pytest

from hypex.dataset import (
    AdditionalTargetRole,
    Dataset,
    ExperimentData,
    GroupingRole,
    TargetRole,
)
from hypex.operators.abstract import GroupOperator
from hypex.utils import BackendsEnum, NotSuitableFieldError


class MeanGap(GroupOperator):
    """Per-group mean of (first target - second target)."""

    @classmethod
    def _inner_function(cls, data, test_data=None, **kwargs):
        return float(data.mean()) - float(test_data.mean())

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields = self._get_fields(data)
        result = self.calc(
            data.ds, group_field=group_field, target_fields=target_fields
        )
        return self._set_value(data, result)


def _ds(df=None, roles=None) -> Dataset:
    df = (
        df
        if df is not None
        else pd.DataFrame(
            {
                "g": ["a", "a", "b", "b"],
                "t1": [1.0, 2.0, 10.0, 20.0],
                "t2": [0.0, 1.0, 4.0, 8.0],
            }
        )
    )
    return Dataset(
        roles=roles or {"g": GroupingRole(), "t1": TargetRole(), "t2": TargetRole()},
        data=df,
        backend=BackendsEnum.pandas,
    )


def _pregrouped(ds):
    df = ds.raw_data
    return [
        ((g,), Dataset(roles=dict(ds.roles), data=part, backend=BackendsEnum.pandas))
        for g, part in df.groupby("g")
    ]


def test_defaults() -> None:
    op = MeanGap()
    assert isinstance(op.target_roles, TargetRole)
    assert isinstance(op.grouping_role, GroupingRole)
    assert op.search_types == [int, float]


def test_custom_roles_are_kept() -> None:
    op = MeanGap(grouping_role=TargetRole(), target_roles=[GroupingRole()])
    assert isinstance(op.grouping_role, TargetRole)
    assert isinstance(op.target_roles[0], GroupingRole)


def test_get_fields_returns_group_and_two_targets() -> None:
    data = ExperimentData(_ds())
    group_field, targets = MeanGap()._get_fields(data)
    assert group_field == ["g"]
    assert sorted(targets) == ["t1", "t2"]


def test_get_fields_adds_additional_target_when_not_two() -> None:
    ds = _ds(
        pd.DataFrame({"g": ["a", "b"], "t1": [1.0, 2.0], "add": [3.0, 4.0]}),
        {"g": GroupingRole(), "t1": TargetRole(), "add": AdditionalTargetRole()},
    )
    _, targets = MeanGap()._get_fields(ExperimentData(ds))
    assert sorted(targets) == ["add", "t1"]


def test_execute_inner_function_dispatches_per_group() -> None:
    ds = _ds()
    result = MeanGap._execute_inner_function(
        _pregrouped(ds), target_fields=["t1", "t2"]
    )
    assert result == {"a": pytest.approx(1.0), "b": pytest.approx(9.0)}


@pytest.mark.parametrize("fields", [None, [], ["t1"], ["t1", "t2", "t3"]])
def test_execute_inner_function_requires_exactly_two_targets(fields) -> None:
    with pytest.raises(ValueError, match="2 targets"):
        MeanGap._execute_inner_function(_pregrouped(_ds()), target_fields=fields)


def test_calc_with_pregrouped_data() -> None:
    ds = _ds()
    result = MeanGap.calc(
        ds, group_field="g", grouping_data=_pregrouped(ds), target_fields=["t1", "t2"]
    )
    assert result == {"a": pytest.approx(1.0), "b": pytest.approx(9.0)}


def test_calc_single_group_raises_not_suitable_field() -> None:
    ds = _ds()
    with pytest.raises(NotSuitableFieldError):
        MeanGap.calc(
            ds,
            group_field="g",
            grouping_data=_pregrouped(ds)[:1],
            target_fields=["t1", "t2"],
        )


def test_set_value_stores_result_in_variables() -> None:
    op = MeanGap()
    data = ExperimentData(_ds())
    stored = op._set_value(data, {"a": 1.0})
    assert stored is data
    assert data.variables[op.id] == {"a": 1.0}


@pytest.mark.xfail(
    strict=True,
    raises=FutureWarning,
    reason="Issue: PandasDataset.count_groups does int(df[cols].nunique()) on a Series, "
    "emitting a FutureWarning, so Dataset.groupby fails under the project's warning filter",
)
def test_calc_groups_data_itself() -> None:
    result = MeanGap.calc(_ds(), group_field="g", target_fields=["t1", "t2"])
    assert result == {"a": pytest.approx(1.0), "b": pytest.approx(9.0)}
