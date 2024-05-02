from abc import abstractmethod, ABC
from typing import Iterable, Any, Union, Callable, Sized, Optional, Dict, Sequence

from hypex.utils import FromDictType


class DatasetBackendNavigation(ABC):

    @property
    def name(self) -> str:
        return str(self.__class__.__name__).lower().replace("backend", "")

    @property
    @abstractmethod
    def index(self):
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @property
    @abstractmethod
    def columns(self):
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def from_dict(
        self, data: FromDictType, index: Optional[Union[Iterable, Sized]] = None
    ):
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def __getitem__(self, item) -> Any:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def _create_empty(
        self,
        index: Optional[Iterable] = None,
        columns: Optional[Iterable[str]] = None,
    ):
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def _get_column_index(
        self, column_name: Union[Sequence[str], str]
    ) -> Union[int, Sequence[int]]:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def _get_column_type(self, column_name: str) -> str:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def _update_column_type(self, column_name: str, type_name: str):
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def add_column(
        self,
        data: Union[Sequence],
        name: str,
        index: Optional[Sequence] = None,
    ):
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def append(self, other, index: bool = False) -> Any:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def loc(self, values: Iterable) -> Iterable:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def iloc(self, values: Iterable) -> Iterable:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )


class DatasetBackendCalc(DatasetBackendNavigation, ABC):
    @abstractmethod
    def mean(self) -> Any:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def max(self) -> Any:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def min(self) -> Any:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def count(self) -> Any:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def sum(self) -> Any:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def agg(self, func) -> Any:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def apply(self, func: Callable, **kwargs) -> Any:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def map(self, func: Callable, **kwargs) -> Any:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def unique(self) -> Any:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def isin(self, values: Iterable) -> Any:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def groupby(self, by: Union[str, Iterable[str]], **kwargs) -> Any:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )
