"""Tests for the ExperimentData blackboard container."""
from __future__ import annotations

import pandas as pd
import pytest

from hypex.dataset import (
    Dataset,
    ExperimentData,
    FeatureRole,
    InfoRole,
    SmallDataset,
    TargetRole,
)
from hypex.utils import ExperimentDataEnum
from hypex.utils.constants import ID_SPLIT_SYMBOL
from hypex.utils.errors import NotFoundInExperimentDataError


def _ed() -> ExperimentData:
    """ExperimentData wrapped around a tiny two-column dataset."""
    ds = Dataset(
        roles={"x": FeatureRole(), "y": TargetRole()},
        data=pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]}),
    )
    return ExperimentData(ds)


def test_init_populates_spaces() -> None:
    """A fresh ExperimentData exposes empty internal spaces."""
    ed = _ed()
    assert ed.variables == {}
    assert ed.groups == {}
    assert ed.analysis_tables == {}
    assert len(ed.ds) == 3


def test_create_empty() -> None:
    """create_empty produces an ExperimentData with an empty dataset."""
    ed = ExperimentData.create_empty(roles={"x": FeatureRole()})
    assert ed.ds.is_empty()


@pytest.mark.parametrize(
    "space",
    [
        ExperimentDataEnum.variables,
        ExperimentDataEnum.additional_fields,
        ExperimentDataEnum.analysis_tables,
        ExperimentDataEnum.groups,
    ],
)
def test_set_value_each_space(space: ExperimentDataEnum) -> None:
    """set_value stores data in every supported space."""
    ed = _ed()

    if space == ExperimentDataEnum.variables:
        ed.set_value(space, "calc", 42, key="result")
        assert ed.variables["calc"]["result"] == 42
    elif space == ExperimentDataEnum.additional_fields:
        ed.set_value(space, "extra", [1, 2, 3], role=InfoRole())
        assert "extra" in ed.ds.columns
    elif space == ExperimentDataEnum.analysis_tables:
        table = SmallDataset.from_dict({"p": [0.01]}, roles=InfoRole())
        ed.set_value(space, "test_id", table)
        assert "test_id" in ed.analysis_tables
    elif space == ExperimentDataEnum.groups:
        group_ds = ed.ds
        ed.set_value(space, "splitter", group_ds, key="control")
        assert "control" in ed.groups["splitter"]


def test_set_value_unknown_space_raises() -> None:
    """Unknown spaces raise ValueError."""
    ed = _ed()
    with pytest.raises(ValueError, match="Unknown space"):
        ed.set_value(ExperimentDataEnum.ml, "x", 1)


@pytest.mark.parametrize("check,result", [("calc", True), ("ghost", False)])
def test_check_hash(check: str, result: bool) -> None:
    """check_hash reports whether an id exists in a space."""
    ed = _ed()
    ed.set_value(ExperimentDataEnum.variables, "calc", 1, key="k")
    assert ed.check_hash(check, ExperimentDataEnum.variables) is result


def test_get_ids_filters_by_class_and_key() -> None:
    """get_ids resolves class names and optional key suffixes."""
    ed = _ed()
    ed.set_value(ExperimentDataEnum.variables, f"MyCalc{ID_SPLIT_SYMBOL}h{ID_SPLIT_SYMBOL}k", 1, key="k")

    found = ed.get_ids("MyCalc", searched_space=ExperimentDataEnum.variables)
    assert len(found["MyCalc"]["variables"]) == 1

    found_keyed = ed.get_ids("MyCalc", searched_space=ExperimentDataEnum.variables, key="k")
    assert len(found_keyed["MyCalc"]["variables"]) == 1


def test_get_one_id_raises_when_absent() -> None:
    """get_one_id raises NotFoundInExperimentDataError for missing ids."""
    ed = _ed()
    with pytest.raises(NotFoundInExperimentDataError):
        ed.get_one_id("MissingClass", ExperimentDataEnum.variables)


@pytest.mark.parametrize(
    "raw,expected",
    [("A", ("A", None)), (f"A{ID_SPLIT_SYMBOL}b", ("A", "b"))],
)
def test_parse_id_for_search(raw: str, expected: tuple) -> None:
    """_parse_id_for_search splits composite ids safely."""
    assert ExperimentData._parse_id_for_search(raw) == expected


def test_copy_is_deep() -> None:
    """copy() produces an independent ExperimentData."""
    ed = _ed()
    clone = ed.copy()
    clone.variables["new"] = {"k": 1}
    assert "new" not in ed.variables


def test_field_search_by_role() -> None:
    """field_search resolves columns by semantic role."""
    ed = _ed()
    assert ed.field_search(FeatureRole()) == ["x"]
    assert ed.field_search(TargetRole()) == ["y"]


def test_field_data_search_returns_subset() -> None:
    """field_data_search returns a Dataset containing matched columns."""
    ed = _ed()
    subset = ed.field_data_search(FeatureRole())
    assert list(subset.columns) == ["x"]