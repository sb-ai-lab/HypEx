"""Tests for StrictABCMeta LSP signature validation."""

from abc import abstractmethod

import pytest

from hypex.utils.strict_abc import StrictABC, StrictABCMeta


class Base(StrictABC):
    @abstractmethod
    def run(self, a, b=1): ...


def test_compatible_override_is_accepted() -> None:
    class Ok(Base):
        def run(self, a, b=1):
            return a + b

    assert Ok().run(1) == 2


def test_child_may_add_defaults() -> None:
    class Ok(Base):
        def run(self, a=0, b=1):
            return a

    assert Ok().run() == 0


def test_removing_default_is_rejected() -> None:
    with pytest.raises(TypeError, match="removing default value for 'b'"):

        class Bad(Base):
            def run(self, a, b):
                return a


def test_removing_default_allowed_when_check_defaults_disabled() -> None:
    class Lax(StrictABC):
        __strict_options__ = {"check_defaults": False}  # noqa: RUF012

        @abstractmethod
        def run(self, a, b=1): ...

    class Child(Lax):
        def run(self, a, b):
            return b

    assert Child().run(1, 2) == 2


def test_parameter_count_mismatch_is_rejected() -> None:
    with pytest.raises(TypeError, match="expected 2 parameters, found 1"):

        class Bad(Base):
            def run(self, a):
                return a


def test_descriptor_mismatch_is_rejected() -> None:
    with pytest.raises(TypeError, match="descriptor type mismatch"):

        class Bad(Base):
            @staticmethod
            def run(a, b=1):
                return a


def test_staticmethod_and_classmethod_are_validated() -> None:
    class S(StrictABC):
        @staticmethod
        @abstractmethod
        def f(a): ...

        @classmethod
        @abstractmethod
        def g(cls, a): ...

    class Ok(S):
        @staticmethod
        def f(a):
            return a

        @classmethod
        def g(cls, a):
            return a

    assert Ok.f(1) == 1 and Ok.g(2) == 2

    with pytest.raises(TypeError, match="expected 1 parameters, found 2"):

        class Bad(S):
            @staticmethod
            def f(a, b):
                return a

            @classmethod
            def g(cls, a):
                return a


def test_narrowing_variadic_is_rejected() -> None:
    class V(StrictABC):
        @abstractmethod
        def f(self, *args, **kwargs): ...

    with pytest.raises(TypeError, match="narrowing variadic"):

        class Bad(V):
            def f(self, a):
                return a


def test_removing_var_positional_or_keyword_is_rejected() -> None:
    class V(StrictABC):
        @abstractmethod
        def f(self, *args, **kwargs): ...

    with pytest.raises(TypeError, match=r"removed \*args"):

        class NoArgs(V):
            def f(self, **kwargs):
                return kwargs

    with pytest.raises(TypeError, match=r"removed \*\*kwargs"):

        class NoKw(V):
            def f(self, *args):
                return args


def test_variadic_preserved_is_accepted_and_adding_variadic_is_accepted() -> None:
    class V(StrictABC):
        @abstractmethod
        def f(self, *args, **kwargs): ...

        @abstractmethod
        def g(self, a): ...

    class Ok(V):
        def f(self, x, *args, **kwargs):
            return x

        def g(self, a, *extra):
            return a

    assert Ok().f(5) == 5 and Ok().g(6) == 6


def test_kind_transition_restricting_is_rejected_expanding_is_accepted() -> None:
    class K(StrictABC):
        @abstractmethod
        def f(self, a, *, b): ...

        @abstractmethod
        def g(self, a, b): ...

    class Ok(K):
        def f(self, a, b):  # keyword-only -> positional-or-keyword is allowed
            return b

        def g(self, a, b):
            return b

    assert Ok().f(1, 2) == 2

    with pytest.raises(TypeError, match="kind transition"):

        class Bad(K):
            def f(self, a, *, b):
                return b

            def g(self, a, *, b):  # positional-or-keyword -> keyword-only
                return b


def test_check_names() -> None:
    class N(StrictABC):
        __strict_options__ = {"check_names": True}  # noqa: RUF012

        @abstractmethod
        def f(self, a): ...

    with pytest.raises(TypeError, match="parameter name mismatch"):

        class Bad(N):
            def f(self, z):
                return z

    class Ok(N):
        def f(self, a):
            return a

    assert Ok().f(3) == 3


def test_check_names_disabled_by_default() -> None:
    class Ok(Base):
        def run(self, x, y=2):
            return x

    assert Ok().run(4) == 4


def test_check_types() -> None:
    class T(StrictABC):
        __strict_options__ = {"check_types": True}  # noqa: RUF012

        @abstractmethod
        def f(self, a: int): ...

    with pytest.raises(TypeError, match="missing type annotation for 'a'"):

        class Missing(T):
            def f(self, a):
                return a

    with pytest.raises(TypeError, match="type annotation mismatch"):

        class Mismatch(T):
            def f(self, a: str):
                return a

    class Ok(T):
        def f(self, a: int):
            return a

    assert Ok().f(1) == 1


def test_check_return_type() -> None:
    class R(StrictABC):
        __strict_options__ = {"check_return_type": True}  # noqa: RUF012

        @abstractmethod
        def f(self) -> int: ...

        @abstractmethod
        def g(self) -> object: ...

    with pytest.raises(TypeError, match="missing return type annotation"):

        class Missing(R):
            def f(self):
                return 1

            def g(self) -> object:
                return 1

    with pytest.raises(TypeError, match="return type not covariant"):

        class NotCov(R):
            def f(self) -> str:
                return "1"

            def g(self) -> object:
                return 1

    class Ok(R):  # bool <= int, int <= object: covariant returns are fine
        def f(self) -> bool:
            return True

        def g(self) -> int:
            return 1

    assert Ok().f() is True


def test_non_class_return_annotation_must_be_equal() -> None:
    class R(StrictABC):
        __strict_options__ = {"check_return_type": True}  # noqa: RUF012

        @abstractmethod
        def f(self) -> "list[int]": ...

    with pytest.raises(TypeError, match="return type not covariant"):

        class Bad(R):
            def f(self) -> "list[str]":
                return []

    class Ok(R):
        def f(self) -> "list[int]":
            return []

    assert Ok().f() == []


def test_uninspectable_signature_is_skipped() -> None:
    class B(StrictABC):
        @abstractmethod
        def f(self): ...

    class Ok(B):
        f = len  # builtin without a Python signature mismatch is tolerated

    assert Ok.f([1, 2]) == 2


def test_abstract_methods_still_enforced_by_abc() -> None:
    with pytest.raises(TypeError, match="abstract"):
        Base()


def test_metaclass_is_exposed() -> None:
    assert isinstance(Base, StrictABCMeta)


def test_child_redeclaring_abstract_method_is_not_validated() -> None:
    class Still(Base):
        @abstractmethod
        def run(self):  # different signature, but still abstract -> skipped
            ...

    with pytest.raises(TypeError, match="abstract"):
        Still()


def test_issubclass_type_error_counts_as_non_covariant() -> None:
    class Picky(type):
        def __subclasscheck__(cls, sub):
            raise TypeError("nope")

    class P(metaclass=Picky):
        pass

    class C:
        pass

    class R(StrictABC):
        __strict_options__ = {"check_return_type": True}  # noqa: RUF012

        @abstractmethod
        def f(self) -> P: ...

    with pytest.raises(TypeError, match="return type not covariant"):

        class Bad(R):
            def f(self) -> C:
                return C()

    class Same(R):  # identical annotation is accepted even when issubclass fails
        def f(self) -> P:
            return P()

    assert isinstance(Same().f(), P)
