from abc import ABC

from hypex.pipelines.base import BaseExecutor


class BaseDataGenerator(ABC, BaseExecutor):
    def __init__(self, num_cols: int):
        self.num_cols = num_cols

    def generate(self):
        pass

    def add(self, df):
        pass


