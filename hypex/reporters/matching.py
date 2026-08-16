from __future__ import annotations

from typing import Any

from ..analyzers.matching import MatchingAnalyzer
from ..dataset import Dataset, ExperimentData
from ..reporters.abstract import Reporter
from ..utils import ExperimentDataEnum, ID_SPLIT_SYMBOL
from ..utils.errors import NotFoundInExperimentDataError


class MatchingReporter(Reporter):
    """Reporter for core matching metrics (ATT, ATC, ATE).

    Retrieves the pre-computed Dataset from either MatchingAnalyzer
    or GroupExperiment.
    """

    def report(self, data: ExperimentData) -> Dataset:
        """Generate the main matching metrics report.

        Attempts to retrieve the results from MatchingAnalyzer first.
        If not found, falls back to GroupExperiment. Returns an empty
        Dataset if neither is present.

        Args:
            data: The experiment data container.

        Returns:
            A Dataset containing the matching metrics, or an empty
            Dataset if no matching results are found.
        """
        try:
            analyzer_id = data.get_one_id(MatchingAnalyzer, ExperimentDataEnum.analysis_tables)
            result_ds = data.analysis_tables[analyzer_id]
            return result_ds
        except NotFoundInExperimentDataError:
            pass

        try:
            group_exp_id = data.get_one_id("GroupExperiment", ExperimentDataEnum.analysis_tables)
            return data.analysis_tables[group_exp_id]
        except NotFoundInExperimentDataError:
            pass

        return Dataset.create_empty()


class MatchingQualityReporter(Reporter):
    """Reporter for matching quality tests (T-Test, Chi2, KS).

    Aggregates the results of OnRoleExperiment from the analysis tables.
    """

    def report(self, data: ExperimentData) -> dict[str, Dataset]:
        """Generate the matching quality tests report.

        Extracts statistical test outcomes from the analysis tables,
        filters out raw statistics, and groups the results by feature name.

        Args:
            data: The experiment data container.

        Returns:
            A dictionary mapping feature names to their corresponding
            quality test result Datasets.
        """
        test_set_for_report = frozenset(["TTest", "Chi2Test", "KSTest", "UTest"])
        quality_results = {}

        for exec_id, table in data.analysis_tables.items():
            is_test = any(test in exec_id for test in test_set_for_report)
            is_raw_stats = exec_id.endswith("stats") or exec_id.endswith("┆stats")

            if is_test and not is_raw_stats:
                parts = exec_id.split(ID_SPLIT_SYMBOL)
                feature_name = parts[-1] if len(parts) > 1 else exec_id

                if feature_name not in quality_results:
                    quality_results[feature_name] = table
                else:
                    quality_results[feature_name] = quality_results[feature_name].append(table)

        return quality_results