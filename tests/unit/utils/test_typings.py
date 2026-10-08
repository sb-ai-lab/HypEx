"""Tests for GenericManager.check_type and SparkTypeMapper."""

from __future__ import annotations

import sys
from decimal import Decimal
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import pytest
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    ByteType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    TimestampType,
)

from hypex.utils.typings import (
    CategoricalTypes,
    DefaultRoleTypes,
    FeatureRoleTypes,
    GenericManager,
    ScalarType,
    SparkTypeMapper,
    TargetRoleTypes,
)

check = GenericManager.check_type


# ---------------------------------------------------------------------------
# check_type: plain types
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "obj,hint,expected",
    [
        (1, int, True),
        ("a", int, False),
        (1.5, float, True),
        (True, int, True),  # bool is an int subclass
        ("a", str, True),
        (None, type(None), True),
        ([1], list, True),
        (object(), Any, True),
        (None, Any, True),
    ],
)
def test_plain_types(obj, hint, expected) -> None:
    assert check(obj, hint) is expected


def test_invalid_type_hint_returns_false() -> None:
    assert check(1, "not a type") is False


# ---------------------------------------------------------------------------
# check_type: unions / optionals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "obj,hint,expected",
    [
        (1, Union[int, str], True),
        ("a", Union[int, str], True),
        (1.5, Union[int, str], False),
        (None, Optional[int], True),
        (3, Optional[int], True),
        ("x", Optional[int], False),
        (2.0, ScalarType, True),
        (b"x", ScalarType, False),
    ],
)
def test_unions(obj, hint, expected) -> None:
    assert check(obj, hint) is expected


# ---------------------------------------------------------------------------
# check_type: parametrised generics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "obj,hint,expected",
    [
        ([1, 2], List[int], True),
        ((1, 2), List[int], False),
        ({"a": 1}, Dict[str, int], True),
        ([1], Dict[str, int], False),
        ({1}, Set[int], True),
        ((1,), Tuple[int, ...], True),
        (frozenset({1}), FrozenSet[int], True),
        # builtin generics (list[int]) exist only on Python >= 3.9
        *(
            [([1], list[int], True), ({"a": 1}, dict[str, int], True)]
            if sys.version_info >= (3, 9)
            else []
        ),
        ([1], List, True),
    ],
)
def test_generics_check_only_the_container(obj, hint, expected) -> None:
    assert check(obj, hint) is expected


def test_union_of_generics() -> None:
    hint = Union[List[int], Dict[str, int]]
    assert check([1], hint) is True
    assert check({"a": 1}, hint) is True
    assert check("s", hint) is False


def test_nested_optional_generic() -> None:
    assert check(None, Optional[List[int]]) is True
    assert check([1], Optional[List[int]]) is True
    assert check(5, Optional[List[int]]) is False


@pytest.mark.xfail(
    strict=True,
    reason="Issue: strict=True is documented to check element types recursively, but both "
    "branches of check_type return True once the container matches",
)
def test_strict_mode_checks_element_types() -> None:
    assert check(["a", "b"], List[int], strict=True) is False


def test_non_strict_mode_ignores_element_types() -> None:
    assert check(["a"], List[int]) is True


# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------
def test_alias_contents() -> None:
    assert CategoricalTypes is str
    assert set(Union[float, int, bool].__args__) == set(TargetRoleTypes.__args__)
    assert set(DefaultRoleTypes.__args__) == {float, bool, str, int}
    assert set(FeatureRoleTypes.__args__) == {float, bool, str, int}
    assert set(ScalarType.__args__) == {float, int, str, bool}


# ---------------------------------------------------------------------------
# SparkTypeMapper
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "spark_type,python_type",
    [
        (IntegerType(), int),
        (LongType(), int),
        (ShortType(), int),
        (ByteType(), int),
        (FloatType(), float),
        (DoubleType(), float),
        (BooleanType(), bool),
        (StringType(), str),
        (DateType(), str),
        (TimestampType(), str),
        (DecimalType(10, 2), Decimal),
    ],
)
def test_spark_type_mapper(spark_type, python_type) -> None:
    assert SparkTypeMapper.to_python(spark_type) is python_type


@pytest.mark.parametrize("spark_type", [BinaryType(), ArrayType(IntegerType())])
def test_unmapped_spark_types_fall_back_to_object(spark_type) -> None:
    assert SparkTypeMapper.to_python(spark_type) is object


def test_mapping_is_read_only() -> None:
    with pytest.raises(TypeError):
        SparkTypeMapper._SPARK_TO_PY[IntegerType] = str  # type: ignore[index]


@pytest.mark.xfail(
    strict=True,
    reason="Issue: to_python is annotated to accept a type-name string, but it looks up "
    "type(spark_type), so strings such as 'long' map to object",
)
def test_string_type_names_are_supported() -> None:
    assert SparkTypeMapper.to_python("long") is int
