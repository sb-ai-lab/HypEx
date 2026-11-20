"""Tests for hypex.utils.enums."""

from __future__ import annotations

import enum
import sys

import pytest
from statsmodels.stats.multitest import multipletests

from hypex.utils import enums
from hypex.utils.enums import (
    ABNTestMethodsEnum,
    ABTestTypesEnum,
    BackendsEnum,
    ExperimentDataEnum,
    RenameEnum,
    SpaceEnum,
)


@pytest.mark.parametrize(
    "enum_cls,members",
    [
        (
            ExperimentDataEnum,
            {"variables", "additional_fields", "analysis_tables", "groups", "ml"},
        ),
        (BackendsEnum, {"pandas", "spark"}),
        (SpaceEnum, {"auto", "additional", "data"}),
        (ABTestTypesEnum, {"t_test", "ks_test", "u_test", "chi2_test"}),
        (RenameEnum, {"all", "columns", "index"}),
        (
            ABNTestMethodsEnum,
            {
                "bonferroni",
                "sidak",
                "holm_sidak",
                "holm",
                "simes_hochberg",
                "hommel",
                "fdr_bh",
                "fdr_by",
                "fdr_tsbh",
                "fdr_tsbky",
                "quantile",
            },
        ),
    ],
)
def test_enum_members(enum_cls, members) -> None:
    assert {m.name for m in enum_cls} == members


@pytest.mark.parametrize(
    "enum_cls",
    [
        ExperimentDataEnum,
        BackendsEnum,
        SpaceEnum,
        ABTestTypesEnum,
        RenameEnum,
        ABNTestMethodsEnum,
    ],
)
def test_enum_values_are_unique_strings_and_roundtrip(enum_cls) -> None:
    values = [m.value for m in enum_cls]
    assert len(values) == len(set(values))
    assert all(isinstance(v, str) for v in values)
    for member in enum_cls:
        assert enum_cls(member.value) is member


def test_all_enums_are_declared_unique() -> None:
    for obj in vars(enums).values():
        if (
            isinstance(obj, type)
            and issubclass(obj, enum.Enum)
            and obj is not enum.Enum
        ):
            # enum.unique raises on duplicates at class creation, so aliases cannot exist
            assert len(obj.__members__) == len(list(obj))


@pytest.mark.parametrize(
    "member",
    [m for m in ABNTestMethodsEnum if m is not ABNTestMethodsEnum.quantile],
    ids=lambda m: m.name,
)
def test_multitest_enum_values_are_valid_statsmodels_methods(member) -> None:
    multipletests([0.01, 0.02, 0.5], method=member.value)


def test_quantile_is_not_a_statsmodels_method() -> None:
    with pytest.raises(ValueError):
        multipletests([0.01, 0.02], method=ABNTestMethodsEnum.quantile.value)


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="Issue: ABTest documents/accepts 'fdr_tsbhy', but the enum (and statsmodels) only "
    "know 'fdr_tsbky' / 'fdr_tsbh'",
)
def test_fdr_tsbhy_is_a_valid_method_name() -> None:
    ABNTestMethodsEnum("fdr_tsbhy")


def test_fdr_tsbky_is_the_registered_name() -> None:
    assert ABNTestMethodsEnum.fdr_tsbky.value == "fdr_tsbky"
    assert "fdr_tsbhy" not in {m.value for m in ABNTestMethodsEnum}


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="PEP 604 annotations need Python 3.10 to evaluate",
)
def test_abtest_literal_matches_enum() -> None:
    """Every method advertised in ABTest's type hint should map to an enum value."""
    import typing

    from hypex import ABTest

    hints = typing.get_type_hints(ABTest.__init__)
    literal = [
        a for a in typing.get_args(hints["multitest_method"]) if a is not type(None)
    ]
    advertised = set(typing.get_args(literal[0]))
    unknown = advertised - {m.value for m in ABNTestMethodsEnum}
    assert unknown == {"fdr_tsbhy"}


def test_enum_members_are_not_interchangeable_with_strings() -> None:
    assert BackendsEnum.pandas != "pandas"
    assert BackendsEnum.pandas.value == "pandas"
