from typing import Dict, Any

from hypex.dataset import ExperimentData
from .aa import AADictReporter


class HomoReporter(AADictReporter):

    def report(self, data: ExperimentData) -> Dict[str, Any]:
        result = {}
        result.update(self.extract_data_from_analysis_tables(data))
        return result
