# hypex/ui/matching.py
from __future__ import annotations

from typing import Any

from ..analyzers.matching import MatchingAnalyzer
from ..dataset import Dataset, ExperimentData, SmallDataset, StatisticRole
from ..extensions.matching_output import MatchingOutputExtension
from ..ml import FaissNearestNeighbors
from ..reporters.matching import MatchingQualityReporter, MatchingReporter
from ..utils import ID_SPLIT_SYMBOL, ExperimentDataEnum
from ..utils.logger import logger
from ..utils.registry import backend_factory
from .base import Output


@logger.log_methods(log_args=False, log_result=False, private=True, static=True)
class MatchingOutput(Output):
    """Output handler for matching experiment results."""

    summary: Dataset
    full_data: Dataset
    quality_results: Dataset

    def __init__(
        self,
        searching_class: type = MatchingAnalyzer,
        extract_full_data: bool = False,
        compute_indexes: bool = False,
    ):
        """Initialize matching output with summary and quality reporters.

        Args:
            searching_class: The analyzer class used to search for results.
            extract_full_data: Whether to build the full matched dataset
                via iterative merges. Set to ``False`` to skip the
                expensive merge + checkpoint loop. Defaults to ``False``.
            compute_indexes: Whether to extract matched indexes.
                Set to ``False`` to skip index collection. Defaults to ``False``.
        """
        super().__init__(
            summary_reporter=MatchingReporter(searching_class, output_format="dict"),
            additional_reporters={"quality_results": MatchingQualityReporter()},
        )
        self.extract_full_data = extract_full_data
        self.compute_indexes = compute_indexes

    @staticmethod
    def _get_extension(experiment_data: ExperimentData) -> MatchingOutputExtension:
        extension_cls = backend_factory.resolve_backend(
            MatchingOutputExtension, experiment_data.ds
        )
        if extension_cls is None:
            raise ValueError(
                f"Matching output is not implemented for {experiment_data.ds.backend_type!r}."
            )
        return extension_cls()

    @staticmethod
    def _get_index_ids(experiment_data: ExperimentData) -> list[str]:
        return experiment_data.get_ids(
            FaissNearestNeighbors, ExperimentDataEnum.additional_fields
        )[FaissNearestNeighbors.__name__][ExperimentDataEnum.additional_fields.value]

    def _extract_full_data(
        self, experiment_data: ExperimentData, indexes: Dataset
    ) -> None:
        """Build full matched data and aligned indexes through the extension."""
        self.full_data, self.indexes = self._get_extension(
            experiment_data
        ).build_full_data(experiment_data, indexes)

    @staticmethod
    def _reformat_summary(summary: dict[str, Any]) -> dict[str, Any]:
        """Reformat a flat summary dictionary with composite keys into a nested structure.

        Args:
            summary: Flat dictionary with composite keys separated by ID_SPLIT_SYMBOL.

        Returns:
            Nested dictionary grouped by metric name and index.
        """
        reformatted_summary: dict[str, Any] = {}
        for key, value in summary.items():
            if ID_SPLIT_SYMBOL not in key:
                continue
            keys = key.split(ID_SPLIT_SYMBOL)
            if keys[0] == "indexes":
                if len(keys) > 2:
                    reformatted_summary.setdefault("indexes", {}).setdefault(
                        keys[1], {}
                    )[keys[2]] = value
                else:
                    reformatted_summary.setdefault("indexes", {})[keys[1]] = value
            else:
                l1_key = keys[0] if len(keys) < 3 else f"{keys[2]} {keys[0]}"
                reformatted_summary.setdefault(l1_key, {})[keys[1]] = value
        return reformatted_summary

    def extract(self, experiment_data: ExperimentData) -> None:
        """Extract and format matching results from experiment data.

        Args:
            experiment_data: The experiment data container with matching
                results stored in ``analysis_tables`` and ``variables``.
        """
        super().extract(experiment_data)
        extension = self._get_extension(experiment_data)
        reformatted_summary = self._reformat_summary(self.summary)

        if self.compute_indexes:
            indexes = extension.extract_indexes(
                experiment_data=experiment_data,
                index_ids=self._get_index_ids(experiment_data),
                summary=self.summary,
                reformatted_summary=reformatted_summary,
            )
        else:
            reformatted_summary.pop("indexes", None)
            indexes = SmallDataset.create_empty()
            logger.debug("Skipping indexes extraction (compute_indexes=False).")

        if reformatted_summary:
            first_key = next(iter(reformatted_summary.keys()))
            group_keys = list(reformatted_summary[first_key].keys())
            transposed_summary = {
                metric: [values[group] for group in group_keys]
                for metric, values in reformatted_summary.items()
            }
            self.summary = SmallDataset.from_dict(
                {"data": transposed_summary, "index": group_keys},
                roles={
                    column: StatisticRole()
                    for column in list(reformatted_summary.keys())
                },
            )
        else:
            self.summary = SmallDataset.create_empty()

        if self.extract_full_data:
            self.full_data, self.indexes = extension.build_full_data(
                experiment_data, indexes
            )
        else:
            self.full_data = Dataset.create_empty(
                roles={},
                backend=experiment_data.ds.backend_type,
                session=experiment_data.ds.session,
            )
            self.indexes = Dataset.create_empty(
                roles={},
                backend=experiment_data.ds.backend_type,
                session=experiment_data.ds.session,
            )
            logger.debug("Skipping full_data extraction (extract_full_data=False).")
        self.summary.raw_data = self.summary.raw_data.round(4)
