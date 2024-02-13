from abc import ABC, abstractmethod
from typing import Iterable

class Executor(ABC):
    @abstractmethod
    def execute(self, data):
        raise NotImplementedError

class Pipeline(Executor):

    def __init__(self, executors: Iterable[Executor]):
        self.executors = executors

    def execute(self, data):
        for executor in self.executors:
            executor.execute(data)