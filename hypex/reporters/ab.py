from __future__ import annotations

from typing import Any, ClassVar

from ..analyzers.ab import ABAnalyzer
from ..comparators import (
    GroupChi2Test,
    GroupKSTest,
    GroupTTest,
    GroupUTest,
    StatsChi2Test,
    StatsKSTest,
    StatsTTest,
    StatsUTest,
)
from ..dataset import ExperimentData
from .abstract import (
    DatasetReporter,
    extract_analyzer_data,
    extract_group_difference,
    extract_group_sizes,
    extract_tests,
)


class ABTestReporter(DatasetReporter):
    """Reporter for A/B test results.

    Extracts group sizes, metric differences, statistical test outcomes,
    and analyzer data, formatting them into a structured dataset or dictionary.
    """

    tests: ClassVar[list] = [
        GroupTTest,
        GroupKSTest,
        GroupUTest,
        GroupChi2Test,
        StatsTTest,
        StatsKSTest,
        StatsChi2Test,
        StatsUTest,
    ]

    def _report(self, data: ExperimentData) -> dict[str, Any]:
        """Generate the final A/B test report.

        Ensures the ``front`` formatting flag is disabled before generating the report,
        then returns the result in the configured format.

        Args:
            data: The experiment data container.

        Returns:
            The report as a dictionary or ``Dataset``.
        """
        result = {}
        result.update(extract_group_sizes(data, self.front))
        result.update(extract_group_difference(data, self.front))
        result.update(extract_tests(data, self.tests, self.front))
        result.update(extract_analyzer_data(data, ABAnalyzer))
        return result
