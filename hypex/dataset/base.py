from abc import ABC, abstractmethod

from hypex.pipelines.base import BaseExecutor


class BaseDataGenerator(ABC, BaseExecutor):
    """
    Base class for data generation.
    Subclasses must implement the following methods.
    """
    def __init__(self, num_cols: int, num_rows: int):
        self.num_cols = num_cols
        self.num_rows = num_rows

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def add(self, df):
        pass


