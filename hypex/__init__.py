from .core import Hypex
from .pipeline import Pipeline

try:
    from ._ipython import load_ipython_extension
    load_ipython_extension()
except ImportError:
    pass

__all__ = ['Hypex', 'Pipeline']