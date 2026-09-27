"""Tests for hypex.dataset.roles — role classes and the role registry."""
from __future__ import annotations

import pytest

from hypex.dataset import (
    ABCRole,
    AdditionalFeatureRole,
    AdditionalGroupingRole,
    AdditionalPreTargetRole,
    AdditionalTargetRole,
    AdditionalTreatmentRole,
    FeatureRole,
    GroupingRole,
    InfoRole,
    PreTargetRole,
    TargetRole,
    TreatmentRole,
    default_roles,
)
from hypex.dataset.roles import (
    AdditionalRole,
    DisabledRole,
    LagRole,
    ReportRole,
    ResumeRole,
    TempRole,
)

# Every concrete role class that must expose a stable, non-empty name.
_ALL_ROLE_CLASSES: tuple[type[ABCRole], ...] = (
    AdditionalFeatureRole,
    AdditionalGroupingRole,
    AdditionalPreTargetRole,
    AdditionalTargetRole,
    AdditionalTreatmentRole,
    DisabledRole,
    FeatureRole,
    GroupingRole,
    InfoRole,
    PreTargetRole,
    ReportRole,
    ResumeRole,
    TargetRole,
    TempRole,
    TreatmentRole,
)

# Roles that have a documented Additional* counterpart in default_roles.
_ADDITIONAL_PAIRS: tuple[tuple[type[ABCRole], type[ABCRole]], ...] = (
    (TargetRole, AdditionalTargetRole),
    (TreatmentRole, AdditionalTreatmentRole),
    (GroupingRole, AdditionalGroupingRole),
    (FeatureRole, AdditionalFeatureRole),
    (PreTargetRole, AdditionalPreTargetRole),
)


@pytest.mark.parametrize("role_cls", _ALL_ROLE_CLASSES)
def test_role_name_for_each_role(role_cls: type[ABCRole]) -> None:
    """Every role class exposes a non-empty canonical role name."""
    role = role_cls()
    assert isinstance(role.role_name, str)
    assert role.role_name


def test_role_names_are_unique_within_registry() -> None:
    """Roles registered in default_roles have pairwise distinct names."""
    names = [r.role_name for r in default_roles.values()]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("data_type", [int, float, str, bool])
def test_astype_sets_data_type(data_type: type) -> None:
    """astype returns a new role carrying the requested data_type."""
    original = FeatureRole()
    casted = original.astype(data_type)

    assert casted.data_type is data_type
    # The source role must not be mutated.
    assert original.data_type is None


def test_copy_is_independent() -> None:
    """copy() produces an independent instance of the same class."""
    from copy import copy

    original = TargetRole(int)
    cloned = copy(original)

    assert cloned is not original
    assert type(cloned) is type(original)

    cloned.data_type = float
    assert original.data_type is int


@pytest.mark.parametrize("base_cls,expected_cls", _ADDITIONAL_PAIRS)
def test_asadditional_maps_to_additional_variant(
    base_cls: type[ABCRole], expected_cls: type[ABCRole]
) -> None:
    """asadditional() resolves registered roles to their Additional variant."""
    result = base_cls().asadditional()
    assert isinstance(result, expected_cls)
    assert isinstance(result, AdditionalRole)


@pytest.mark.parametrize(
    "role_cls", [DisabledRole, ResumeRole, TempRole, ReportRole]
)
def test_asadditional_returns_same_class_for_unregistered(
    role_cls: type[ABCRole],
) -> None:
    """Roles without a registered Additional variant keep their own class."""
    result = role_cls().asadditional()
    assert type(result) is role_cls


def test_default_roles_keys_are_lowercase_and_resolve() -> None:
    """default_roles keys are lowercase strings mapping to ABCRole instances."""
    assert len(default_roles) > 0
    for key, role in default_roles.items():
        assert isinstance(key, str)
        assert key == key.lower()
        assert isinstance(role, ABCRole)


def test_pretarget_role_carries_lag_and_cofounders() -> None:
    """PreTargetRole stores lag, parent and cofounders metadata."""
    role = PreTargetRole(parent="spend", lag=1, cofounders=["age", "gender"])

    assert role.parent == "spend"
    assert role.lag == 1
    assert role.cofounders == ["age", "gender"]


def test_lag_role_subclasses() -> None:
    """FeatureRole and PreTargetRole belong to the LagRole family."""
    assert issubclass(FeatureRole, LagRole)
    assert issubclass(PreTargetRole, LagRole)


def test_star_import_exposes_index_role() -> None:
    """Regression: hypex.dataset.__all__ must expose IndexRole (Phase 0.5)."""
    import hypex.dataset as dataset_module

    assert "IndexRole" in dataset_module.__all__
    assert "indexRole" not in dataset_module.__all__