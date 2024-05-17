from typing import Dict, Any

from hypex.dataset import ExperimentData
from .aa import AADictReporter


class HomoReporter(AADictReporter):

    def report(self, data: ExperimentData) -> Dict[str, Any]:
        return self.extract_data_from_analysis_tables(data)
