from __future__ import annotations

import datetime
from types import MappingProxyType
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
    Type, 
    Union
)
from pyspark.sql.types import (
    DataType,
    StringType,
    LongType,
    DoubleType,
    BooleanType,
    DecimalType,
    DateType,
    TimestampType,
    ArrayType,
    StructType,
    MapType,
    BinaryType,
    IntegerType,
    FloatType,
    ShortType,
    ByteType,
)

import numpy as np
import pandas as pd
import pyspark.sql as spark
import pyspark.pandas as ps


if TYPE_CHECKING:
    from hypex.dataset import Dataset

StratificationRoleTypes = float | str | datetime.datetime
DefaultRoleTypes = float | bool | str | int
TargetRoleTypes = float | int | bool
FeatureRoleTypes = float | bool | str | int
CategoricalTypes = str
ScalarType = float | int | str | bool
PysparkScalarType = (
    IntegerType | LongType |
    FloatType | DoubleType |
    DecimalType | ShortType | ByteType
)
GroupingDataType = tuple[list[tuple[str, "Dataset"]], list[tuple[str, "Dataset"]]]
SourceDataTypes = pd.DataFrame | ps.DataFrame | spark.DataFrame


MultiFieldKeyTypes = str | Sequence[str]

FromDictTypes = (
    dict[str, list[Any]] |
    list[Dict[Any, Any]] |
    dict[str, Dict[Any, List]] |
    dict[str, "Dataset"]
)
RoleNameType = str
DecoratedType = TypeVar("DecoratedType", bound=Union[Callable[..., Any], property])
DocstringInheritDecorator = Callable[[DecoratedType], DecoratedType]

SetParamsDictTypes = dict[str, Any] | dict[type, dict[str, Any]]

class SparkTypeMapper:
    _SPARK_TO_PY: MappingProxyType[type[DataType], type] = MappingProxyType({
        IntegerType: int, 
        LongType: int, 
        ShortType: int, 
        ByteType: int,
        FloatType: float, 
        DoubleType: float,
        BooleanType: bool,
        StringType: str,
        DateType: str, 
        TimestampType: str,
    })

    @classmethod
    def to_python(cls, spark_type: DataType | str) -> type:
        return cls._SPARK_TO_PY.get(type(spark_type), object)