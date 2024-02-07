from abc import ABC, abstractmethod


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
