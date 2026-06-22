from __future__ import annotations
import warnings
from ..comparators import GroupChi2Test, GroupDifference, GroupSizes, GroupKSTest, GroupTTest
from .abstract import DatasetReporter, DictReporter, extract_group_sizes, extract_group_difference, extract_tests

class HomogeneityReporter(DatasetReporter):
    def _report(self, data: ExperimentData) -> dict:
        result = {}
        result.update(extract_group_sizes(data, self.front))
        result.update(extract_group_difference(data, self.front))
        result.update(extract_tests(data, [GroupTTest, GroupKSTest, GroupChi2Test], self.front))
        return result

class HomoDictReporter(HomogeneityReporter):
    def __init__(self, front=True):
        super().__init__(DictReporter(front=front), output_format="dict")
        warnings.warn("HomoDictReporter is deprecated.", DeprecationWarning, stacklevel=2)

class HomoDatasetReporter(HomogeneityReporter):
    def __init__(self):
        super().__init__(DictReporter(), output_format="dataset")
        warnings.warn("HomoDatasetReporter is deprecated.", DeprecationWarning, stacklevel=2)