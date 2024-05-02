from abc import ABC, abstractmethod
from typing import Iterable


# TODO 1: реализовать каскад абстракций (Хранение+Навигация -> Расчеты)
# TODO 2: реализовать каскад абстракций для бэкенда
# TODO 3: магические методы операторов
# TODO 4: математичекие функции (abs, log, mean, median, mode, variance, std, min, max, _shuffle, unique, corr, transpose)
# TODO 5: Enum для статистических тестов + реализация выбора функции из словаря
# TODO 6: для фильтров (n_na, n_unique, value_counts, quantile, select_dtype, join, drop)
# TODO 7: для предобработки (fillna, dropna, sort)


class DatasetBaseNavigation(ABC):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def _create_empty(self, index=None):
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def add_column(self, data, name):
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

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


class DatasetBase(DatasetBaseNavigation, ABC):
    @abstractmethod
    def apply(self, *args, **kwargs):
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def map(self, *args, **kwargs):
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def unique(self):
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def isin(self, values: Iterable) -> Iterable[bool]:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )

    @abstractmethod
    def groupby(self, by, **kwargs):
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )


class DatasetBackend(DatasetBase):

    @property
    def name(self):
        return str(self.__class__.__name__).lower().replace("dataset", "")

    @abstractmethod
    def loc(self, values: Iterable) -> Iterable:
        return

    @abstractmethod
    def iloc(self, values: Iterable) -> Iterable:
        raise NotImplementedError(
            "This method is abstract and will be overridden in DatasetBase class."
        )
