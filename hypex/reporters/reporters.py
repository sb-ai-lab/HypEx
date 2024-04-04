from abc import ABC, abstractmethod
from typing import Dict

from hypex.dataset.dataset import ExperimentData

class Reporter(ABC):
    @abstractmethod
    def report(self, data: ExperimentData):
        raise NotImplementedError

class DictReporter(Reporter, ABC):
    @abstractmethod
    def report(self, data: ExperimentData) -> Dict:
        raise NotImplementedError