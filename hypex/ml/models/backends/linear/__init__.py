from ....._spark import PYSPARK_AVAILABLE
from .pandas import PandasLinearModel

if PYSPARK_AVAILABLE:
    from .spark import SparkLinearModel
