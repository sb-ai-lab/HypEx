from abc import abstractmethod
from typing import Iterable


class DatasetBackend:

    @property
    def name(self):
        return str(self.__class__.__name__).lower().replace("dataset", "")

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
