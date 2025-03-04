import datetime
from typing import Any, Union, Dict, List, TypeVar, Callable, Sequence, Tuple

StratificationRoleTypes = Union[float, str, datetime.datetime]
DefaultRoleTypes = Union[float, bool, str, int]
TargetRoleTypes = Union[float, int, bool]
CategoricalTypes = Union[str]
ScalarType = Union[float, int, str, bool]
GroupingDataType = Tuple[List[Tuple[str, "Dataset"]], List[Tuple[str, "Dataset"]]]


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
