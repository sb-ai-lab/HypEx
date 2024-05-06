import datetime
from typing import Any, Union, Dict, List, TypeVar, Callable

StratificationRoleTypes = Union[float, str, datetime.datetime]
DefaultRoleTypes = Union[float, bool, str, int]
TargetRoleTypes = Union[float, int, bool]
CategoricalTypes = str

FieldKeyTypes = Union[str, int]

FromDictType = Union[Dict[str, List[Any]], List[Dict[Any, Any]]]
RoleNameType = str
DecoratedType = TypeVar("DecoratedType", bound=Union[Callable[..., Any], property])
DocstringInheritDecorator = Callable[[DecoratedType], DecoratedType]
