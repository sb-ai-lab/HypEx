"""Tests for inherit_docstring_from."""

from __future__ import annotations

import pytest

from hypex.utils.decorator import inherit_docstring_from


def source_func():
    """Source docs."""


def test_decorates_callable_and_preserves_behavior() -> None:
    @inherit_docstring_from(source_func)
    def target(x, y=2):
        """Own docs."""
        return x * y

    assert target.__doc__ == "Source docs."
    assert target.__name__ == "target"
    assert target(3) == 6
    assert target(3, y=4) == 12


def test_decorates_property() -> None:
    class A:
        @property
        @inherit_docstring_from(source_func)
        def value(self):
            return 7

    assert A.value.__doc__ == "Source docs."
    assert A().value == 7


def test_property_keeps_setter_and_deleter() -> None:
    store = {}

    prop = property(
        lambda self: store.get("v"),
        lambda self, v: store.__setitem__("v", v),
        lambda self: store.clear(),
    )
    new = inherit_docstring_from(source_func)(prop)
    assert isinstance(new, property)
    assert new.fget is prop.fget and new.fset is prop.fset and new.fdel is prop.fdel
    assert new.__doc__ == "Source docs."


def test_source_without_docstring_gives_none_doc() -> None:
    def nodoc():
        pass

    @inherit_docstring_from(nodoc)
    def target():
        """Own."""

    assert target.__doc__ is None


def test_non_callable_non_property_raises() -> None:
    with pytest.raises(TypeError, match="callables or properties"):
        inherit_docstring_from(source_func)(42)
