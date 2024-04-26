from abc import ABC, abstractmethod
from typing import Iterable


class DatasetBase(ABC):

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def add_column(self, data, name):
        pass

    def from_dict(self, data):
        pass

    def _create_empty(self, index=None):
        pass

    @abstractmethod
    def apply(self, *args, **kwargs):
        pass

    @abstractmethod
    def map(self, *args, **kwargs):
        pass

    @abstractmethod
    def unique(self):
        pass

    @property
    @abstractmethod
    def index(self):
        pass

    @property
    @abstractmethod
    def columns(self):
        pass

    @abstractmethod
    def isin(self, values: Iterable) -> Iterable[bool]:
        pass

    def groupby(self, by, **kwargs):
        pass

    def loc(self, values: Iterable) -> Iterable:
        pass

    def iloc(self, values: Iterable) -> Iterable:
        pass


class DatasetBackend(DatasetBase):
    @property
    def name(self):
        return str(self.__class__.__name__).lower().replace("dataset", "")
