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

class SparkTypeMapper:
    _PY_TO_SPARK: Dict[Type, DataType] = {
        # Strings
        str: StringType(),
        bytes: BinaryType(),
        # Integers (order matters: bool is subclass of int!)
        bool: BooleanType(),
        int: LongType(),
        np.integer: LongType(),
        np.int64: LongType(),
        np.int32: IntegerType(),
        np.int16: ShortType(),
        np.int8: ByteType(),
        # Floats
        float: DoubleType(),
        np.floating: DoubleType(),
        np.float64: DoubleType(),
        np.float32: FloatType(),
        # Date/Time
        # datetime.date: DateType(),
        # datetime.datetime: TimestampType(),
        # Special
        type(None): StringType(),
        object: StringType(),
    }

    @classmethod
    def to_spark(cls, value: Any | Type) -> DataType:
        if isinstance(value, type):
            return cls._resolve_python_type(value)
        return cls._resolve_python_value(value)

    @classmethod
    def _resolve_python_type(cls, py_type: Type) -> DataType:
        if py_type in cls._PY_TO_SPARK:
            return cls._PY_TO_SPARK[py_type]
        
        for base_type, spark_type in cls._PY_TO_SPARK.items():
            if isinstance(base_type, type) and issubclass(py_type, base_type):
                return spark_type
        
        return StringType()

    @classmethod
    def _resolve_python_value(cls, value: Any) -> DataType:
        if value is None:
            return StringType()
        
        py_type = type(value)
        
        if isinstance(value, bool):
            return BooleanType()
        
        if isinstance(value, np.integer):
            if isinstance(value, (np.int8, np.int16)):
                return ShortType()
            elif isinstance(value, np.int32):
                return IntegerType()
            else:
                return LongType()
        
        if isinstance(value, np.floating):
            if isinstance(value, np.float32):
                return FloatType()
            else:
                return DoubleType()
        
        # Стандартные типы
        return cls._resolve_python_type(py_type)

    _SPARK_TO_PY: Dict[str, Type] = {
        # Integers
        "tinyint": int,
        "byte": int,
        "smallint": int,
        "short": int,
        "int": int,
        "integer": int,
        "bigint": int,
        "long": int,
        # Floats
        "float": float,
        "real": float,
        "double": float,
        "decimal": float,
        "numeric": float,
        # Strings
        "string": str,
        "varchar": str,
        "char": str,
        "text": str,
        # Boolean
        "boolean": bool,
        "bool": bool,
        # Date/Time → str
        "date": str,
        "timestamp": str,
        "timestamp_ntz": str,
        # Complex
        "array": list,
        "struct": dict,
        "map": dict,
        "object": object,
        # Binary
        "binary": bytes,
        "blob": bytes,
        # Null
        "null": type(None),
    }

    @classmethod
    def to_python(cls, spark_type: DataType | str) -> Type:
        if isinstance(spark_type, DataType):
            type_str = spark_type.simpleString()
        elif isinstance(spark_type, str):
            type_str = spark_type.strip().lower()
        else:
            raise TypeError(
                f"spark_type must be DataType or str, got {type(spark_type).__name__}"
            )
        
        base_type = type_str.split("(")[0].split("<")[0].strip()
        
        return cls._SPARK_TO_PY.get(base_type, object)

    @classmethod
    def get_spark_type_name(cls, spark_type: DataType) -> str:
        return spark_type.simpleString().split("(")[0]

    @classmethod
    def is_numeric(cls, spark_type: DataType | str) -> bool:
        py_type = cls.to_python(spark_type)
        return py_type in (int, float)

    @classmethod
    def is_string(cls, spark_type: DataType | str) -> bool:
        py_type = cls.to_python(spark_type)
        return py_type is str

    @classmethod
    def is_complex(cls, spark_type: DataType | str) -> bool:
        type_str = spark_type.simpleString() if isinstance(spark_type, DataType) else str(spark_type)
        base = type_str.split("(")[0].split("<")[0].strip().lower()
        return base in ("array", "struct", "map")

    @classmethod
    def validate_cast(cls, from_type: DataType | str, to_type: DataType | str) -> bool:
        from_py = cls.to_python(from_type)
        to_py = cls.to_python(to_type)
        
        if to_py is str:
            return True
        
        if from_py is str and to_py in (int, float):
            return True
        
        if from_py in (int, float) and to_py in (int, float):
            return True
        
        if from_py is bool and to_py in (int, float):
            return True
        if from_py in (int, float) and to_py is bool:
            return True
        
        return from_py == to_py