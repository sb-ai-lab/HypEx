from .encoders import DummyEncoderExtension
from .faiss import FaissExtension
from .scipy_linalg import CholeskyExtension, InverseExtension
from .scipy_stats import (
    GroupChi2TestExtension,
    GroupKSTestExtension,
    GroupTTestExtension,
    GroupUTestExtension,
)
from .stats_hypothesis_testing import (   # ← НОВОЕ
    StatsAggregationExtension,
    StatsKSTestExtension,
    StatsChi2TestExtension,
)
from .statsmodels import MultiTest, MultitestQuantile

__all__ = [
    "GroupChi2TestExtension",
    "CholeskyExtension",
    "DummyEncoderExtension",
    "FaissExtension",
    "InverseExtension",
    "GroupKSTestExtension",
    "MultiTest",
    "MultitestQuantile",
    "GroupTTestExtension",
    "GroupUTestExtension",
    "StatsAggregationExtension",
    "StatsKSTestExtension",
    "StatsChi2TestExtension",
]
