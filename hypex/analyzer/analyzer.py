from abc import ABC
from hypex.experiment.experiment import ComplexExecutor

class Analyzer(ComplexExecutor, ABC):
    def calc(self, data: Dataset) -> Dataset:
        warnings.warn(f"Meaningless for {self.__class__.__name__}")
        return data