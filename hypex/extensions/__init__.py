from .encoders import DummyEncoderExtension
try:
    from .faiss import FaissExtension
except ImportError:
    FaissExtension = None
from .scipy_linalg import CholeskyExtension, InverseExtension
from .scipy_stats import (
                          Chi2TestExtension,
                          KSTestExtension,
                          TTestExtension,
                          UTestExtension,
)
from .statsmodels import MultiTest, MultitestQuantile

__all__ = [
    "Chi2TestExtension",
    "CholeskyExtension",
    "DummyEncoderExtension",
    "FaissExtension",
    "InverseExtension",
    "KSTestExtension",
    "MultiTest",
    "MultitestQuantile",
    "TTestExtension",
    "UTestExtension",
]
