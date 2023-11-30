from abc import ABC, abstractmethod

from hypex.utils.typings import ExecutorSequence


class BaseExecutor(ABC):
    @abstractmethod
    def execute(self, **kwargs):
        pass


class BasePipeline(ABC, BaseExecutor):
    def __init__(self, executors: ExecutorSequence):
        self.executors = executors

    def execute(self):
        for executor in self.executors:
            executor.execute()
