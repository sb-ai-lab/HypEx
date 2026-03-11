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
from typing import TYPE_CHECKING, Any, TypeVar, Callable, Sequence

import numpy as np
import pandas as pd
import pyspark.sql as spark
import pyspark.pandas as ps


if TYPE_CHECKING:
    from hypex.dataset import Dataset

StratificationRoleTypes = Union[float, str, datetime.datetime]
DefaultRoleTypes = Union[float, bool, str, int]
TargetRoleTypes = Union[float, int, bool]
FeatureRoleTypes = Union[float, bool, str, int]
CategoricalTypes = str
ScalarType = Union[float, int, str, bool]
PysparkScalarType = (
    IntegerType, LongType,
    FloatType, DoubleType,
    DecimalType, ShortType, ByteType
)
GroupingDataType = Tuple[List[Tuple[str, "Dataset"]], List[Tuple[str, "Dataset"]]]
SourceDataTypes = Union[pd.DataFrame, ps.DataFrame, spark.DataFrame]


MultiFieldKeyTypes = Union[str, Sequence[str]]

FromDictTypes = Union[
    Dict[str, List[Any]],
    List[Dict[Any, Any]],
    Dict[str, Dict[Any, List]],
    Dict[str, "Dataset"]
]
RoleNameType = str
DecoratedType = TypeVar("DecoratedType", bound=Union[Callable[..., Any], property])
DocstringInheritDecorator = Callable[[DecoratedType], DecoratedType]

SetParamsDictTypes = Union[Dict[str, Any], Dict[type, Dict[str, Any]]]

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
    def to_python(cls, spark_type: Union[DataType, str]) -> type:
        return cls._SPARK_TO_PY.get(type(spark_type), object)