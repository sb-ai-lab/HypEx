from abc import ABC, abstractmethod
from typing import Union, List

from hypex.dataset import Dataset


class Task(ABC):
    @abstractmethod
    def calc(self, data: Dataset, **kwargs):
        pass


class CompareTask(Task):
    @abstractmethod
    def calc(self, data: Dataset, other: Union[Dataset, List[Dataset], None] = None, **kwargs):
        pass
