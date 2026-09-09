"""Backend factory for automatic selection of backend-specific implementations.

Factory is a design pattern used to create objects without specifying
their concrete classes. In the context of HypEx, the factory maps a
"master" class (e.g., ``FaissExtension``) to a concrete implementation
(e.g., ``PandasFaissExtension``, ``SparkFaissExtension``) based on the
backend of the input data.

Typical usage:

    >>> from hypex.utils.registry import backend_factory
    >>>
    >>> @backend_factory.register(MasterClass, PandasDataset)
    ... class PandasImpl(MasterClass):
    ...     ...
    >>>
    >>> resolved_cls = backend_factory.resolve_backend(MasterClass, data)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dataset import Dataset

__all__ = ["BackendFactory", "backend_factory"]


class BackendFactory:
    """Registry that maps ``(base_class, backend_type)`` to an implementation class.

    The factory enables runtime dispatch: given a master/abstract class and a
    dataset instance, it resolves the concrete backend-specific implementation
    registered for that combination.

    Attributes:
        _registry: Internal mapping ``{base_cls: {backend_type: impl_cls}}``.

    Example:
        >>> factory = BackendFactory()
        >>>
        >>> @factory.register(BaseEncoder, PandasDataset)
        ... class PandasEncoder(BaseEncoder):
        ...     pass
        >>>
        >>> cls = factory.resolve_backend(BaseEncoder, pandas_data)
        >>> assert cls is PandasEncoder
    """

    def __init__(self) -> None:
        self._registry: dict[type, dict[type, type]] = {}

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_backends(
        backend_types: type | Iterable[type],
    ) -> list[type]:
        """Normalize a single type or an iterable of types into a flat list.

        Args:
            backend_types: One backend type or an iterable of backend types.

        Returns:
            A list containing all provided backend types.
        """
        if isinstance(backend_types, (list, tuple, set, frozenset)):
            return list(backend_types)
        return [backend_types]

    def _insert(
        self,
        base_cls: type,
        backend_types: type | Iterable[type],
        impl_cls: type,
    ) -> None:
        """Insert an implementation for every listed backend type.

        Args:
            base_cls: The master/abstract class to register against.
            backend_types: Backend type(s) that trigger this implementation.
            impl_cls: The concrete implementation class to store.
        """
        for b_type in self._normalize_backends(backend_types):
            self._registry.setdefault(base_cls, {})[b_type] = impl_cls

    # ── Registration API ─────────────────────────────────────────────────────

    def register(
        self,
        base_cls: type,
        backend_types: type | Iterable[type],
    ):
        """Decorator to register a backend-specific implementation.

        Supports a single backend type or an iterable of types. The decorated
        class is returned unchanged; registration is a side-effect.

        Args:
            base_cls: The master/abstract class to register against.
            backend_types: Backend type(s) that trigger this implementation.

        Returns:
            A decorator that registers the class and returns it unchanged.

        Example:
            >>> @backend_factory.register(MasterClass, PandasDataset)
            ... class PandasImpl(MasterClass):
            ...     pass

            >>> @backend_factory.register(MasterClass, [PandasDataset, SparkDataset])
            ... class UniversalImpl(MasterClass):
            ...     pass
        """

        def decorator(cls: type) -> type:
            self._insert(base_cls, backend_types, cls)
            return cls

        return decorator

    def register_explicit(
        self,
        base_cls: type,
        backend_types: type | Iterable[type],
        impl_cls: type,
    ) -> None:
        """Register an implementation without using decorator syntax.

        Useful when the implementation class is defined in a separate module
        or generated dynamically.

        Args:
            base_cls: The master/abstract class to register against.
            backend_types: Backend type(s) that trigger this implementation.
            impl_cls: The concrete implementation class to register.

        Example:
            >>> backend_factory.register_explicit(
            ...     MasterClass, [PandasDataset, SparkDataset], UniversalImpl
            ... )
        """
        self._insert(base_cls, backend_types, impl_cls)

    def unregister(self, base_cls: type, backend_type: type | None = None) -> bool:
        """Remove a registration from the factory.

        Args:
            base_cls: The master class to unregister from.
            backend_type: Specific backend type to remove. If ``None``,
                all backends registered for ``base_cls`` are removed.

        Returns:
            ``True`` if at least one entry was removed, ``False`` otherwise.

        Example:
            >>> backend_factory.unregister(MasterClass, PandasDataset)
            True
            >>> backend_factory.unregister(MasterClass)
            True
        """
        backends = self._registry.get(base_cls)
        if backends is None:
            return False

        if backend_type is None:
            del self._registry[base_cls]
            return True

        return backends.pop(backend_type, None) is not None

    # ── Resolution API ───────────────────────────────────────────────────────

    def resolve_backend(self, base_cls: type, data: Dataset) -> type | None:
        """Resolve the backend-specific implementation for a given base class.

        Inspects ``type(data.backend_data)`` and looks up the corresponding
        registered subclass in the factory registry.

        Args:
            base_cls: The master/abstract class to resolve
                (e.g., ``FaissExtension``, ``DummyEncoderExtension``).
            data: The dataset instance whose underlying backend type
                (``type(data.backend_data)``) is used as the lookup key.

        Returns:
            The backend-specific subclass if a match is found in the registry.
            Returns ``None`` if no backends are registered for ``base_cls``,
            indicating that the base class itself should be used as a fallback.

        Example:
            >>> # Assuming 'data' is backed by Pandas
            >>> backend_cls = backend_factory.resolve_backend(FaissExtension, data)
            >>> if backend_cls is not None:
            ...     instance = backend_cls(n_neighbors=5)
        """
        cls_backends = self._registry.get(base_cls)
        if not cls_backends:
            return None

        backend_type = type(data.backend_data)
        return cls_backends.get(backend_type)

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def registry(self) -> dict[type, dict[type, type]]:
        """Return a shallow copy of the registry (read-only snapshot).

        Returns:
            A dictionary with the same structure as the internal registry.
            Mutating the returned dict does not affect the factory state.
        """
        return {k: dict(v) for k, v in self._registry.items()}

    def print_registry(self) -> None:
        """Pretty-print the current registry structure to stdout.

        Outputs each registered base class and its backend-to-implementation
        mappings.

        Example:
            >>> backend_factory.print_registry()
            Key class – FaissExtension:
                Backend – PandasDataset → PandasFaissExtension
                Backend – SparkDataset → SparkFaissExtension
        """
        for base_cls, backends in self._registry.items():
            print(f"Key class – {base_cls.__name__}:")
            for backend, impl in backends.items():
                print(f"\tBackend – {backend.__name__} → {impl.__name__}")

    # Backward-compatible alias (typo preserved for existing callers).
    rigestry_output = print_registry

    def __repr__(self) -> str:
        """Return a concise string representation for debugging.

        Returns:
            A string showing the number of registered base classes
            and total registrations.
        """
        total = sum(len(v) for v in self._registry.values())
        return (
            f"<BackendFactory bases={len(self._registry)} "
            f"registrations={total}>"
        )


# Singleton — single instance shared across the entire process.
backend_factory = BackendFactory()
