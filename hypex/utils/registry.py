from __future__ import annotations
from typing import (
    Dict, 
    Type,
    Union,
    Iterable
)
# from ..dataset import Dataset

"""
Factory — это паттерн проектирования, который используется для создания объектов 
без указания конкретных классов объектов. 
"""
class BackendFactory:
    """
    Backend-factory class for automatic selection of backend-dependency class realization.
    It selects direct realization due to input data backend.
    """

    def __init__(self):
        self._registry: Dict[Type, Dict[Type, Type]] = {}
    
    def register(self, base_cls: Type, backend_types: Union[Type, Iterable[Type]]):
        """
        Decorator to register a backend-specific implementation.
        Supports single type or iterable of types.

        Usage:
        ```python 
            @backend_factory.register(MasterBackendClass, PandasDataset)
            class BackendDependendClass(...):
                ...
        ```
        Usage: 
        ```python 
            @backend_factory.register(MasterBackendClass, [PandasDataset, SparkDataset])
            class BackendDependendClass(...):
                ...
        ```
        """
        def decorator(cls: Type):
            backends = backend_types if isinstance(backend_types, (list, tuple, set)) else [backend_types]
            for b_type in backends:
                self._registry.setdefault(base_cls, {})[b_type] = cls
            return cls
        return decorator

    def register_explicit(self, base_cls: Type, backend_types: Union[Type, Iterable[Type]], impl_cls: Type):
        """Explicit registration without decorator. Supports single type or iterable."""
        backends = backend_types if isinstance(backend_types, (list, tuple, set)) else [backend_types]
        for b_type in backends:
            self._registry.setdefault(base_cls, {})[b_type] = impl_cls

    @property
    def registry(self):
        return self._registry
    
    def rigestry_output(self):
        """
        Print `backend_factory` structure.
        """
        for key, value in self._registry.items():
            print(f"Key class - {key.__name__}:")
            for backend, realization in value.items():
                print(f"\tBackend - {backend.__name__} : realization - {realization.__name__}")
    
    def resolve_backend(self, base_cls: Type, data):
        """
        Resolve and return the backend-specific implementation class for a given base class.

        This method inspects the backend type of the provided ``data`` (e.g., ``PandasDataset``
        or ``SparkDataset``) and looks up the corresponding registered subclass for the
        specified ``base_cls`` in the factory registry.

        Args:
            base_cls (Type): The master or abstract base class to resolve 
                (e.g., ``FaissExtension``, ``DummyEncoderExtension``).
            data (Dataset): The dataset instance. Its underlying backend type 
                (``type(data.backend_data)``) is used as the lookup key in the registry.

        Returns:
            Type | None: The backend-specific subclass if a match is found in the registry. 
            Returns ``None`` if the ``base_cls`` has no registered backends, indicating 
            that the base class itself should be used as a fallback.

        Example:
            ```python
                # Assuming 'data' is backed by Pandas
                backend_cls = backend_factory.resolve_backend(FaissExtension, data)
                
                # backend_cls will resolve to PandasFaissExtension
                if backend_cls is not None:
                    instance = backend_cls(n_neighbors=5)
            ```
        """
        cls_backends = self._registry.get(base_cls)
        backend_type = type(data.backend_data)

        if not cls_backends:
            # raise NotImplementedError(f"{base_cls.__name__} doesn't exist!")
            return None # no such class in factory, so base_cls is what we need

        cls = cls_backends.get(backend_type)

        return cls

# Singleton
backend_factory = BackendFactory()


    