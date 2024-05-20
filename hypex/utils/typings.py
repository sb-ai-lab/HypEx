import datetime
from typing import Any, Union, Dict, List, TypeVar, Callable, Sequence, Literal

StratificationRoleTypes = Union[float, str, datetime.datetime]
DefaultRoleTypes = Union[float, bool, str, int]
TargetRoleTypes = Union[float, int, bool]
CategoricalTypes = str
ScalarType = Union[float, int, str, bool]

FieldKeyTypes = Union[str, int]
MultiFieldKeyTypes = Union[FieldKeyTypes, Sequence[FieldKeyTypes]]

FromDictTypes = Union[Dict[str, List[Any]], List[Dict[Any, Any]]]
ABNTestMethodsTypes = Literal[
    "bonferroni",
    "sidak",
    "holm-sidak",
    "holm",
    "simes-hochberg",
    "hommel",
    "fdr_bh",
    "fdr_by",
    "fdr_tsbh",
    "fdr_tsbky",
]
RoleNameType = str
DecoratedType = TypeVar("DecoratedType", bound=Union[Callable[..., Any], property])
DocstringInheritDecorator = Callable[[DecoratedType], DecoratedType]
