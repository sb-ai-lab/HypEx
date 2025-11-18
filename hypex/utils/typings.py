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

if TYPE_CHECKING:
    from hypex.dataset import Dataset

StratificationRoleTypes = Union[float, str, datetime.datetime]
DefaultRoleTypes = Union[float, bool, str, int]
TargetRoleTypes = Union[float, int, bool]
FeatureRoleTypes = Union[float, bool, str, int]
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
