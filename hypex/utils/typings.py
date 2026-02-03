import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from pyspark.sql.types import (
    IntegerType, LongType, FloatType,
    DoubleType, DecimalType, ShortType, ByteType
)

import pandas as pd
import pyspark.sql as spark

if TYPE_CHECKING:
    from hypex.dataset import Dataset

StratificationRoleTypes = Union[float, str, datetime.datetime]
DefaultRoleTypes = Union[float, bool, str, int]
TargetRoleTypes = Union[float, int, bool]
FeatureRoleTypes = Union[float, bool, str, int]
CategoricalTypes = Union[str]
ScalarType = Union[float, int, str, bool]
PysparkScalarType = Union[
    IntegerType, LongType,
    FloatType, DoubleType,
    DecimalType, ShortType, ByteType
]
GroupingDataType = Tuple[List[Tuple[str, "Dataset"]], List[Tuple[str, "Dataset"]]]
SourceDataTypes = Union[pd.DataFrame, spark.DataFrame]


MultiFieldKeyTypes = Union[str, Sequence[str]]

FromDictTypes = Union[
    Dict[str, List[Any]],
    List[Dict[Any, Any]],
    Dict[str, Dict[Any, List]],
    Dict[str, "Dataset"],
]
RoleNameType = str
DecoratedType = TypeVar("DecoratedType", bound=Union[Callable[..., Any], property])
DocstringInheritDecorator = Callable[[DecoratedType], DecoratedType]

SetParamsDictTypes = Union[Dict[str, Any], Dict[type, Dict[str, Any]]]
