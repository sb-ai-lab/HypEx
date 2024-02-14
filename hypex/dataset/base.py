from abc import ABC, abstractmethod
from typing import Iterable


class DatasetBase(ABC):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, key, value):
        raise NotImplementedError

    @abstractmethod
    def create_empty(self, indexes=None, columns=None):
        raise NotImplementedError

    @abstractmethod
    def apply(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def map(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def unique(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def index(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def columns(self):
        raise NotImplementedError

    @abstractmethod
    def isin(self, values: Iterable) -> Iterable[bool]:
        raise NotImplementedError

    @abstractmethod
    def groupby(self):
        raise NotImplementedError
