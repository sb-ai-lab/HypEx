from abc import ABC, abstractmethod

# TODO: needed functions:
"""
index getter

"""

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

    # TODO
    @abstractmethod
    def unique():
        pass

    # TODO
    @abstractmethod
    @property
    def index():
        pass

    # TODO
    @abstractmethod
    def isin(values:Iterable)->Iterable[bool]:
        pass

    # TODO: тут надо разобраться с сигнатурой
    @abstractmethod
    def groupby():
        pass
