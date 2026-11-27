from .abstract import MLModel
from .backends import linear as _linear_backends
from .linear import LinearModel

__all__ = ["LinearModel", "MLModel"]
