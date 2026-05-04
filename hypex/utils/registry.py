from __future__ import annotations
from typing import (
    Dict, 
    Type,
    Union,
    Iterable
)
# from ..dataset import Dataset

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
        Usage: @backend_factory.register(BaseComparator, PandasDataset)
        Usage: @backend_factory.register(BaseComparator, [PandasDataset, SparkDataset])
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
        Get realization of class depending on data backend type.
        """
        cls_backends = self._registry.get(base_cls)
        backend_type = type(data.backend_data)

        if not cls_backends:
            # raise NotImplementedError(f"{base_cls.__name__} doesn't exist!")
            return None # no such class in factory, so base_cls is what we need

        cls = cls_backends.get(backend_type)

        # TODO: deside how to work with cases when there are no realizations for that backend
        # if not cls:
        #     supported = [b.__name__ for b in cls_backends.keys()]
        #     raise NotImplementedError(
        #         f"{base_cls.__name__} does not support {backend_type.__name__}. "
        #         f"Available backends: {', '.join(supported)}"
        #     )

        return cls

# Singleton
backend_factory = BackendFactory()


    