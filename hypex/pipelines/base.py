from abc import ABC

from hypex.utils.typings import ExecutorSequence


class BaseExecutor(ABC):
    def execute(self, **kwargs):
        pass


class BasePipeline(ABC, BaseExecutor):
    def __init__(self, executors: ExecutorSequence):
        self.executors = executors
