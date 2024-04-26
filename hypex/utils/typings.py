import datetime
from typing import Any, Union, Dict, List, TypeVar, Callable

DefaultRoleTypes = Union[float, bool, str, int]
TargetRoleTypes = Union[float, int, bool]
CategoricalTypes = str

FieldKey = Union[str, int]

FromDictType = Union[Dict[str, List[Any]], List[Dict[Any, Any]]]

DecoratedType = TypeVar('DecoratedType', bound=Union[Callable[..., Any], property])
DocstringInheritDecorator = Callable[[DecoratedType], DecoratedType]
