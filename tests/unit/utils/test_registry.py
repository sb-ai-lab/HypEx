"""Tests for BackendFactory register/resolve."""

from __future__ import annotations

import pandas as pd
import pytest

from hypex.dataset import Dataset, FeatureRole
from hypex.dataset.backends import PandasDataset, SparkDataset
from hypex.utils import BackendsEnum
from hypex.utils.registry import BackendFactory, backend_factory


class Master:
    pass


class OtherMaster:
    pass


@pytest.fixture
def factory() -> BackendFactory:
    return BackendFactory()


@pytest.fixture
def pandas_ds() -> Dataset:
    return Dataset(
        roles={"x": FeatureRole()},
        data=pd.DataFrame({"x": [1.0]}),
        backend=BackendsEnum.pandas,
    )


def test_decorator_registers_and_returns_class_unchanged(factory) -> None:
    @factory.register(Master, PandasDataset)
    class Impl(Master):
        pass

    assert Impl.__name__ == "Impl"
    assert factory.registry == {Master: {PandasDataset: Impl}}


def test_resolve_returns_registered_impl(factory, pandas_ds) -> None:
    @factory.register(Master, PandasDataset)
    class Impl(Master):
        pass

    assert factory.resolve_backend(Master, pandas_ds) is Impl


def test_resolve_unregistered_base_returns_none(factory, pandas_ds) -> None:
    assert factory.resolve_backend(Master, pandas_ds) is None


def test_resolve_missing_backend_for_registered_base_returns_none(
    factory, pandas_ds
) -> None:
    @factory.register(Master, SparkDataset)
    class SparkImpl(Master):
        pass

    assert factory.resolve_backend(Master, pandas_ds) is None


@pytest.mark.parametrize("container", [list, tuple, set, frozenset])
def test_register_multiple_backends(factory, pandas_ds, container) -> None:
    @factory.register(Master, container([PandasDataset, SparkDataset]))
    class Universal(Master):
        pass

    assert factory.registry[Master] == {
        PandasDataset: Universal,
        SparkDataset: Universal,
    }
    assert factory.resolve_backend(Master, pandas_ds) is Universal


def test_register_explicit(factory, pandas_ds) -> None:
    class Impl(Master):
        pass

    factory.register_explicit(Master, PandasDataset, Impl)
    assert factory.resolve_backend(Master, pandas_ds) is Impl


def test_later_registration_overrides_earlier(factory, pandas_ds) -> None:
    @factory.register(Master, PandasDataset)
    class First(Master):
        pass

    @factory.register(Master, PandasDataset)
    class Second(Master):
        pass

    assert factory.resolve_backend(Master, pandas_ds) is Second


def test_registrations_for_different_bases_are_independent(factory, pandas_ds) -> None:
    @factory.register(Master, PandasDataset)
    class A(Master):
        pass

    @factory.register(OtherMaster, PandasDataset)
    class B(OtherMaster):
        pass

    assert factory.resolve_backend(Master, pandas_ds) is A
    assert factory.resolve_backend(OtherMaster, pandas_ds) is B


def test_unregister_single_backend(factory) -> None:
    factory.register_explicit(Master, [PandasDataset, SparkDataset], Master)
    assert factory.unregister(Master, PandasDataset) is True
    assert list(factory.registry[Master]) == [SparkDataset]
    assert factory.unregister(Master, PandasDataset) is False


def test_unregister_whole_base(factory) -> None:
    factory.register_explicit(Master, [PandasDataset, SparkDataset], Master)
    assert factory.unregister(Master) is True
    assert Master not in factory.registry
    assert factory.unregister(Master) is False


def test_unregister_unknown_base_is_false(factory) -> None:
    assert factory.unregister(Master) is False


def test_registry_property_is_a_copy(factory) -> None:
    factory.register_explicit(Master, PandasDataset, Master)
    snapshot = factory.registry
    snapshot[Master].clear()
    snapshot.clear()
    assert factory.registry == {Master: {PandasDataset: Master}}


def test_print_registry_and_alias(factory, capsys) -> None:
    class Impl(Master):
        pass

    factory.register_explicit(Master, PandasDataset, Impl)
    factory.print_registry()
    printed = capsys.readouterr().out
    assert "Key class – Master" in printed
    assert "PandasDataset → Impl" in printed
    assert BackendFactory.rigestry_output is BackendFactory.print_registry


def test_repr_counts_registrations(factory) -> None:
    factory.register_explicit(Master, [PandasDataset, SparkDataset], Master)
    factory.register_explicit(OtherMaster, PandasDataset, OtherMaster)
    assert repr(factory) == "<BackendFactory bases=2 registrations=3>"


def test_global_factory_resolves_library_extensions(pandas_ds) -> None:
    from hypex.extensions import FaissExtension, PandasFaissExtension

    assert (
        backend_factory.resolve_backend(FaissExtension, pandas_ds)
        is PandasFaissExtension
    )


@pytest.mark.spark
def test_global_factory_resolves_spark_extension(spark_session) -> None:
    from hypex.extensions import FaissExtension, SparkFaissExtension

    ds = Dataset(
        roles={"x": FeatureRole()},
        data=pd.DataFrame({"x": [1.0]}),
        backend=BackendsEnum.spark,
        session=spark_session,
    )
    assert backend_factory.resolve_backend(FaissExtension, ds) is SparkFaissExtension
