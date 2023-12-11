from abc import ABC, abstractmethod

from hypex.utils.typings import ExecutorSequence


class BaseExecutor(ABC):
    """
    Abstract base executor.
    Parent class for all classes with execution. It defines main functions to implement one structure.
    """
    @abstractmethod
    def execute(self, **kwargs):
        pass


class BasePipeline(BaseExecutor):
    """
    Abstract base pipline.
    Defines structure to execute special pipelines.
    """
    def __init__(self, executors: ExecutorSequence):
        self.executors = executors

    def execute(self):
        for executor in self.executors:
            executor.execute()
