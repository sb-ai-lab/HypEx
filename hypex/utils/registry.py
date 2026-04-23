from __future__ import annotations
from typings import Dict, Type
from ..dataset import Dataset

class BackendRegistry:
    """
    Backend register class.
    """

    def __init__(self):
        self._registry: Dict[Type, Dict[Type, Type]] = {}
    
    def registry(self, base_cls: Type, backend_type: Type):
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
    
    def resolve_backend(self, base_cls: Type, data: Dataset):
        """
        Get realization of class depending on data backend type.
        """
        cls_backends = self._registry.get(base_cls)
        backend_type = type(data.backend)

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
backend_registry = BackendRegistry()


    