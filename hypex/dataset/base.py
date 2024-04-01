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
    def add_column(self, data, name):
        raise NotImplementedError

    def from_dict(self, data):
        raise NotImplementedError

    @abstractmethod
    def _create_empty(self, index=None, columns=None):
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

    def groupby(self, by, **kwargs):
        raise NotImplementedError

    def loc(self, values: Iterable) -> Iterable:
        raise NotImplementedError

    def iloc(self, values: Iterable) -> Iterable:
        raise NotImplementedError
