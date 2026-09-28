from __future__ import annotations

import warnings
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
)
from ..dataset import Dataset, ExperimentData, SmallDataset, StatisticRole
from ..dataset.experiment_data import ExperimentDataEnum
from .abstract import (
    DatasetReporter,
    DictReporter,
    Reporter,
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
        GroupTTest, GroupKSTest, GroupUTest, GroupChi2Test,
        StatsTTest, StatsKSTest, StatsChi2Test,
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

    @staticmethod
    def report_variance_reductions(data: ExperimentData) -> Dataset | str:
        """Extract variance reduction metrics from CUPED results in analysis_tables.

        Args:
            data: The experiment data container.

        Returns:
            A ``SmallDataset`` with columns ``Transformed Metric Name``
            and ``Variance Reduction (%)``, or a descriptive string if
            no variance reduction data is available.
        """
        from ..transformers.cuped import CUPEDTransformer

        ids = data.get_ids(
            CUPEDTransformer,
            searched_space=ExperimentDataEnum.analysis_tables,
        )
        table_ids = ids.get(CUPEDTransformer.__name__, {}).get(
            ExperimentDataEnum.analysis_tables.value, [],
        )
        if not table_ids:
            return (
                "No variance reduction data available. "
                "Ensure CUPED or CUPAC was applied."
            )

        table = data.analysis_tables[table_ids[0]]
        if table.is_empty():
            return "No variance reduction data available."

        records = table.to_records()
        report_data = [
            {
                "Transformed Metric Name": row["feature"],
                "Variance Reduction (%)": row["variance_reduction_pct"],
            }
            for row in records
        ]

        return SmallDataset.from_dict(
            report_data,
            roles={
                "Transformed Metric Name": StatisticRole(str),
                "Variance Reduction (%)": StatisticRole(float),
            },
        )

class ABDictReporter(ABTestReporter):
    """Legacy reporter wrapper for dictionary output.

    Deprecated: Use ``ABTestReporter(output_format='dict')`` instead.
    """
    def __init__(self, front: bool = True):
        """Initialize the legacy dictionary reporter.

        Args:
            front: If ``True``, formats keys for front-end display.
                Defaults to ``True``.
        """
        super().__init__(DictReporter(front=front), output_format="dict")
        warnings.warn("ABDictReporter is deprecated.", DeprecationWarning, stacklevel=2)

class ABDatasetReporter(ABTestReporter):
    """Legacy reporter wrapper for dataset output.
    Deprecated: Use ``ABTestReporter()`` instead.
    """
    def __init__(self):
        """Initialize the legacy dataset reporter."""
        super().__init__(
            DictReporter(),
            output_format="dataset",
            invert_pass=True,  # AB: significant effect = OK
        )
        warnings.warn("ABDatasetReporter is deprecated.", DeprecationWarning, stacklevel=2)
