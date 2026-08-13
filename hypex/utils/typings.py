from __future__ import annotations

import datetime
import typing
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
    Type,
    Union,
    get_args,
    get_origin
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

class GenericManager:

    TYPE_MAP ={
        typing.List: list,
        typing.Dict: dict,
        typing.Set: set, 
        typing.Tuple: tuple,
        typing.FrozenSet: frozenset 
    }

    @staticmethod
    def check_type(obj: object, type_hint: Any, strict: bool=False) -> bool:
        """
        Check object type. Supports parametric jenerics.
        Works from python 3.8+.

        Args
        ----
            obj: `object`
                object for type check;

            type_hint: `Any`
                type_hint;
            
            strict: `bool`
                If `True` checks recurentli all types.
                If `False` (by default), checks only external type.
        """

        origin = get_origin(type_hint)
        args = get_args(type_hint)

        # no generic
        if origin is None:
            if type_hint is Any:
                return True
            try:
                return isinstance(obj, type_hint)
            except TypeError:
                return False

        # Union ot Optional type
        if origin is Union:
            return any(GenericManager.check_type(obj, arg, strict) for arg in args)
        
        # type correction for python types
        base_type = origin
        if hasattr(base_type, '__origin__'):
            base_type = base_type.__origin__

        base_type = GenericManager.TYPE_MAP.get(base_type, base_type)

        if not isinstance(obj, base_type):
            return False
        
        if not strict or not args:
            return True

        return True

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