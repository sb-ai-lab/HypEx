from abc import ABC, abstractmethod

from hypex.pipelines.base import BaseExecutor


class BaseDataGenerator(ABC, BaseExecutor):
    def __init__(self, num_cols: int):
        self.num_cols = num_cols

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def add(self, df):
        pass


