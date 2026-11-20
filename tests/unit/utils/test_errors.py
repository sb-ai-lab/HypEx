"""Tests for all exception classes in hypex.utils.errors."""

from __future__ import annotations

import inspect

import pytest

from hypex.utils import errors
from hypex.utils.errors import (
    AbstractMethodError,
    BackendTypeError,
    ConcatBackendError,
    ConcatDataError,
    DataTypeError,
    InvalidArgumentError,
    MergeOnError,
    NoColumnsError,
    NoneArgumentError,
    NoRequiredArgumentError,
    NotFoundInExperimentDataError,
    NotSuitableFieldError,
    PairsNotFoundError,
    RoleColumnError,
    SpaceError,
)

CASES = [
    (RoleColumnError, ("roles", "cols"), ["roles", "cols", "Check your roles"]),
    (ConcatDataError, (list,), ["Can only append Dataset to Dataset", "list"]),
    (ConcatBackendError, ("spark", "pandas"), ["same backends", "spark", "pandas"]),
    (SpaceError, ("weird",), ["weird", "not a valid space"]),
    (NoColumnsError, ("Target",), ["Target", "No columns found"]),
    (NotSuitableFieldError, ("g", "Grouping"), ["Grouping field g", "not suitable"]),
    (NotFoundInExperimentDataError, ("Foo",), ["Foo", "not found"]),
    (DataTypeError, (int,), ["Dataset and Dataset", "int"]),
    (BackendTypeError, ("A", "B"), ["same backends", "Got A expected B"]),
    (MergeOnError, ("col",), ["merge", "col"]),
    (NoRequiredArgumentError, ("x",), ["required argument x"]),
    (NoneArgumentError, ("x", "fit"), ["Argument x is None", "fit"]),
    (InvalidArgumentError, ("x", "int"), ["Invalid type for argument x", "int"]),
    (PairsNotFoundError, (), ["Pairs are not found"]),
    (AbstractMethodError, (), ["abstract"]),
]


@pytest.mark.parametrize(
    "exc_cls,args,fragments", CASES, ids=[c[0].__name__ for c in CASES]
)
def test_error_message_contains_context(exc_cls, args, fragments) -> None:
    error = exc_cls(*args)
    for fragment in fragments:
        assert fragment in str(error)


@pytest.mark.parametrize("exc_cls,args,_", CASES, ids=[c[0].__name__ for c in CASES])
def test_error_can_be_raised_and_caught(exc_cls, args, _) -> None:
    with pytest.raises(exc_cls):
        raise exc_cls(*args)


def test_abstract_method_error_is_not_implemented_error() -> None:
    assert issubclass(AbstractMethodError, NotImplementedError)
    with pytest.raises(NotImplementedError):
        raise AbstractMethodError()


def test_all_other_errors_derive_from_exception_only() -> None:
    for exc_cls, _, _ in CASES:
        if exc_cls is AbstractMethodError:
            continue
        assert issubclass(exc_cls, Exception)
        assert not issubclass(exc_cls, NotImplementedError)


def test_every_error_class_in_module_is_covered() -> None:
    defined = {
        name
        for name, obj in vars(errors).items()
        if inspect.isclass(obj)
        and issubclass(obj, Exception)
        and obj.__module__ == errors.__name__
    }
    assert defined == {c[0].__name__ for c in CASES}


NOT_EXPORTED = {"NoneArgumentError", "InvalidArgumentError", "PairsNotFoundError"}


def test_most_errors_are_exported_from_utils_package() -> None:
    import hypex.utils as utils

    for exc_cls, _, _ in CASES:
        if exc_cls.__name__ in NOT_EXPORTED:
            continue
        assert getattr(utils, exc_cls.__name__) is exc_cls


@pytest.mark.xfail(
    strict=True,
    reason="Issue: NoneArgumentError, InvalidArgumentError and PairsNotFoundError are not "
    "re-exported from hypex.utils (callers must import hypex.utils.errors)",
)
def test_all_errors_are_exported_from_utils_package() -> None:
    import hypex.utils as utils

    for exc_cls, _, _ in CASES:
        assert getattr(utils, exc_cls.__name__) is exc_cls


def test_role_column_error_formats_both_collections() -> None:
    message = str(RoleColumnError(["a"], ["b", "c"]))
    assert "['a']" in message and "['b', 'c']" in message
