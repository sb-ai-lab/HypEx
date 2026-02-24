from __future__ import annotations

import datetime
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import pandas as pd
import pyspark.sql as spark
from pyspark.sql.types import (
    ByteType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
)

if TYPE_CHECKING:
    from hypex.dataset import Dataset

StratificationRoleTypes = float | str | datetime.datetime
DefaultRoleTypes = float | bool | str | int
TargetRoleTypes = float | int | bool
FeatureRoleTypes = float | bool | str | int
CategoricalTypes = str
ScalarType = float | int | str | bool
PysparkScalarType = (
    IntegerType | LongType | FloatType | DoubleType | DecimalType | ShortType | ByteType
)
GroupingDataType = tuple[list[tuple[str, "Dataset"]], list[tuple[str, "Dataset"]]]
SourceDataTypes = pd.DataFrame | spark.DataFrame


MultiFieldKeyTypes = str | Sequence[str]

FromDictTypes = (
    dict[str, list[Any]]
    | list[dict[Any, Any]]
    | dict[str, dict[Any, list]]
    | dict[str, "Dataset"]
)
RoleNameType = str
DecoratedType = TypeVar("DecoratedType", bound=Callable[..., Any] | property)
DocstringInheritDecorator = Callable[[DecoratedType], DecoratedType]

SetParamsDictTypes = dict[str, Any] | dict[type, dict[str, Any]]

class SparkTypeMapper:
    @staticmethod
    def types(value):
        if value is None:
            return StringType()
        elif isinstance(value, str):
            return StringType()
        elif isinstance(value, int):
            return LongType()
        elif isinstance(value, float):
            return DoubleType()
        else:
            return StringType()
