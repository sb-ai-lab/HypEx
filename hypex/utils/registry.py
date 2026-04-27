from __future__ import annotations
from typing import Dict, Type
# from ..dataset import Dataset

class BackendFactory:
    """
    Backend-factory class for automatic selection of backend-dependency class realization.
    It selects direct realization due to input data backend.
    """

    def __init__(self):
        self._registry: Dict[Type, Dict[Type, Type]] = {}
    
    def register(self, base_cls: Type, backend_type: Type):
        """
        Decorator to register a backend-specific implementation.
        Usage: @registry.register(FaissExtention, PandasDataset)
        """
        def decorator(cls: Type):
            self._registry.setdefault(base_cls, {})[backend_type] = cls
            return cls
        return decorator

    def register_explicit(self, base_cls: Type, backend_type: Type, impl_cls: Type):
        """Explicit registration without decorator (useful for dynamic loading)."""
        self._registry.setdefault(base_cls, {})[backend_type] = impl_cls
    
    @property
    def registry(self):
        return self._registry
    
    def resolve_backend(self, base_cls: Type, data):
        """
        Get realization of class depending on data backend type.
        """
        cls_backends = self._registry.get(base_cls)
        backend_type = type(data.backend_data)

        if not cls_backends:
            raise NotImplementedError(f"{base_cls.__name__} doesn't exist!")

        cls = cls_backends.get(backend_type)

        if not cls:
            supported = [b.__name__ for b in cls_backends.keys()]
            raise NotImplementedError(
                f"{base_cls.__name__} does not support {backend_type.__name__}. "
                f"Available backends: {', '.join(supported)}"
            )

        return cls

# Singleton
backend_factory = BackendFactory()


    