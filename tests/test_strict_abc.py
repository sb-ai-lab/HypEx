import pytest
from abc import abstractmethod

from hypex.utils import StrictABCMeta

class TestStrictABCMeta:
    def test_valid_basic_implementation(self):
        class Animal(metaclass=StrictABCMeta):
            @abstractmethod
            def speak(self, volume: float = 1.0): pass
        class Dog(Animal):
            def speak(self, loudness: float = 1.0): pass
        assert callable(Dog().speak)

    def test_defaults_lsp_compliance(self):
        class BaseAddDefault(metaclass=StrictABCMeta):
            __strict_options__ = {'check_defaults': True}
            @abstractmethod
            def run(self, speed: int): pass
        class FastRunner(BaseAddDefault):
            def run(self, speed: int = 10): pass
        assert isinstance(FastRunner(), BaseAddDefault)

        class BaseRemoveDefault(metaclass=StrictABCMeta):
            __strict_options__ = {'check_defaults': True}
            @abstractmethod
            def walk(self, pace: int = 5): pass
        with pytest.raises(TypeError) as exc_info:
            class SlowWalker(BaseRemoveDefault):
                def walk(self, pace: int): pass
        assert "strengthens precondition" in str(exc_info.value)

    def test_invalid_extra_parameter(self):
        class Base(metaclass=StrictABCMeta):
            @abstractmethod
            def jump(self, height: float): pass
        with pytest.raises(TypeError) as exc_info:
            class HighJumper(Base):
                def jump(self, height: float, style: str): pass
        assert "expected 1 parameters, found 2" in str(exc_info.value)

    def test_invalid_type_annotation(self):
        class Base(metaclass=StrictABCMeta):
            __strict_options__ = {'check_types': True}
            @abstractmethod
            def feed(self, amount: int): pass
        with pytest.raises(TypeError) as exc_info:
            class BadFeeder(Base):
                def feed(self, amount: float): pass
        assert "type annotation mismatch" in str(exc_info.value)

    def test_return_type_covariance(self):
        class Animal: pass
        class Dog(Animal): pass

        class Base(metaclass=StrictABCMeta):
            __strict_options__ = {'check_return_type': True}
            @abstractmethod
            def get_companion(self) -> Animal: pass

        class GoodChild(Base):
            def get_companion(self) -> Dog: pass
        assert isinstance(GoodChild(), Base)

        with pytest.raises(TypeError) as exc_info:
            class BadChild(Base):
                def get_companion(self) -> str: pass
        assert "return type not covariant" in str(exc_info.value)

    def test_kind_transition_lsp(self):
        class Base(metaclass=StrictABCMeta):
            @abstractmethod
            def call(self, x, /): pass

        class SafeExpansion(Base):
            def call(self, x): pass
        assert isinstance(SafeExpansion(), Base)

        with pytest.raises(TypeError) as exc_info:
            class UnsafeRestriction(Base):
                def call(self, *, x): pass
        assert "restricts call syntax" in str(exc_info.value)

    def test_non_standard_receiver_name(self):
        class Base(metaclass=StrictABCMeta):
            @classmethod
            @abstractmethod
            def create(klass, config: dict): pass

        class Impl(Base):
            @classmethod
            def create(cls, config: dict): pass
        assert isinstance(Impl(), Base)

    def test_staticmethod_descriptor_mismatch(self):
        class Base(metaclass=StrictABCMeta):
            @staticmethod
            @abstractmethod
            def compute(x, y): pass
        with pytest.raises(TypeError) as exc_info:
            class Impl(Base):
                def compute(self, x, y): pass
        assert "descriptor type mismatch" in str(exc_info.value)

    def test_variadic_lsp_safe(self):
        class Base(metaclass=StrictABCMeta):
            @abstractmethod
            def handle(self, a: int, b: str): pass
        class Handler(Base):
            def handle(self, *args, **kwargs): pass
        assert isinstance(Handler(), Base)

    def test_variadic_lsp_violation(self):
        class Base(metaclass=StrictABCMeta):
            @abstractmethod
            def process(self, *args, **kwargs): pass
        with pytest.raises(TypeError) as exc_info:
            class NarrowImpl(Base):
                def process(self, x: int): pass
        assert "violates LSP" in str(exc_info.value)

    def test_variadic_kind_replacement_violation(self):
        class Base(metaclass=StrictABCMeta):
            @abstractmethod
            def run(self, *args): pass
        with pytest.raises(TypeError) as exc_info:
            class BadImpl(Base):
                def run(self, **kwargs): pass
        assert "removed *args" in str(exc_info.value)

        class Base2(metaclass=StrictABCMeta):
            @abstractmethod
            def walk(self, **kwargs): pass
        with pytest.raises(TypeError) as exc_info:
            class BadWalk(Base2):
                def walk(self, *args): pass
        assert "removed **kwargs" in str(exc_info.value)