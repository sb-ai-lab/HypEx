import warnings

# ── Suppress noisy warnings at library level ────────────────────────────────
# PySpark Pandas API advice warnings ("index_col not specified for to_spark")
try:
    from pyspark.pandas.utils import PandasAPIOnSparkAdviceWarning
    warnings.filterwarnings("ignore", category=PandasAPIOnSparkAdviceWarning)
except ImportError:
    pass  # PySpark not installed — nothing to suppress
# ─────────────────────────────────────────────────────────────────────────────

from .__version__ import __version__
from .aa import AATest
from .ab import ABTest
from .homogeneity import HomogeneityTest
from .matching import Matching

__all__ = ["AATest", "ABTest", "HomogeneityTest", "Matching", "__version__"]