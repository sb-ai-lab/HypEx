from __future__ import annotations

from ..comparators import (
    GroupChi2Test,
    GroupKSTest,
    GroupTTest,
)
from ..dataset import ExperimentData
from .abstract import (
    DatasetReporter,
    extract_group_difference,
    extract_group_sizes,
    extract_tests,
)


class HomogeneityReporter(DatasetReporter):
    def _report(self, data: ExperimentData) -> dict:
        result = {}
        result.update(extract_group_sizes(data, self.front))
        result.update(extract_group_difference(data, self.front))
        result.update(
            extract_tests(data, [GroupTTest, GroupKSTest, GroupChi2Test], self.front)
        )
        return result
