from .encoders import DummyEncoderExtension
# from .faiss import FaissExtension
from .faiss_exploded import FaissExtension, SparkFaissExtension, PandasFaissExtension
from .scipy_linalg import CholeskyExtension, InverseExtension
# from .scipy_stats import (
#     GroupChi2TestExtension,
#     GroupKSTestExtension,
#     GroupTTestExtension,
#     GroupUTestExtension,
# )
from .scipy_stats import (
    GroupChi2TestExtension,
    GroupKSTestExtension,
    GroupTTestExtension,
    GroupUTestExtension,
    PandasTTestExtension,
    PandasKSTestExtension,
    PandasUTestExtension,
    PandasChi2TestExtension,
    SparkTTestExtension,
    SparkKSTestExtension,
    SparkUTestExtension,
    SparkChi2TestExtension
)

from .statsmodels import MultiTest, MultitestQuantile

# __all__ = [
#     "GroupChi2TestExtension",
#     "CholeskyExtension",
#     "DummyEncoderExtension",
#     "FaissExtension",
#     "InverseExtension",
#     "GroupKSTestExtension",
#     "MultiTest",
#     "MultitestQuantile",
#     "GroupTTestExtension",
#     "GroupUTestExtension",
# ]
__all__ = [
    "DummyEncoderExtension",
    "FaissExtension",
    "SparkFaissExtension",
    "PandasFaissExtension",
    "CholeskyExtension",
    "InverseExtension",
    # "PandasGroupStatTest",
    # "SparkGroupStatTest",
    "GroupTTestExtension",
    "GroupUTestExtension",
    "GroupChi2TestExtension",
    "GroupKSTestExtension",
    "PandasTTestExtension",
    "PandasKSTestExtension",
    "PandasUTestExtension",
    "PandasChi2TestExtension",
    "SparkTTestExtension",
    "SparkKSTestExtension",
    "SparkUTestExtension",
    "SparkChi2TestExtension",
    "MultiTest",
    "MultitestQuantile",
]