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
    def __setitem__(self, key, value):
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
    def index(self):
        return

    @abstractmethod
    def isin(self, values: Iterable) -> Iterable[bool]:
        pass

    @abstractmethod
    def groupby(self):
        pass
