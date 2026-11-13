# hypex/ui/matching.py
from __future__ import annotations

import warnings
from typing import Any

from ..analyzers.matching import MatchingAnalyzer
from ..dataset import (
    AdditionalMatchingRole,
    Dataset,
    ExperimentData,
    GroupingRole,
    IndexRole,
    InfoRole,
    SmallDataset,
    StatisticRole,
)
from ..ml import FaissNearestNeighbors
from ..reporters.matching import MatchingQualityReporter, MatchingReporter
from ..utils import (
    ID_SPLIT_SYMBOL,
    MATCHING_INDEXES_SPLITTER_SYMBOL,
    BackendsEnum,
    ExperimentDataEnum,
)
from ..utils.adapter import Adapter
from ..utils.logger import logger
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

    def _extract_full_data(
        self, experiment_data: ExperimentData, indexes: Dataset
    ) -> None:
        """Build full matched data and aligned indexes through Dataset lookups."""
        backend = experiment_data.ds.backend_type
        if indexes.backend_type != backend:
            indexes = indexes.to_backend(
                backend=backend,
                session=experiment_data.ds.session,
            )

        if not experiment_data.ds.is_persisted:
            experiment_data.ds.checkpoint(eager=True)
        original_columns = set(experiment_data.ds.columns)
        ds_reset = experiment_data.ds.reset_index(drop=False)
        idx_col = next(c for c in ds_reset.columns if c not in original_columns)
        ds_reset.checkpoint(eager=True)

        self.indexes = Dataset.create_empty(
            roles={},
            backend=backend,
            session=experiment_data.ds.session,
        )
        full_data = None

        for i in range(len(indexes.columns)):
            t_indexes = indexes.iloc[:, i]
            col_name = t_indexes.columns[0]
            matched_data = ds_reset.lookup_by_index(
                t_indexes, col_name, source_index_column=idx_col
            )
            matched_data = matched_data.rename(
                {col: f"{col}_matched_{i}" for col in matched_data.columns}
            )

            reindexed_matched = experiment_data.ds.merge(
                matched_data,
                left_index=True,
                right_index=True,
                how="left",
            )
            reindexed_matched = reindexed_matched.drop(
                columns=list(experiment_data.ds.columns),
            )

            if self.indexes.is_empty():
                self.indexes = t_indexes
            else:
                self.indexes = self.indexes.add_column(
                    data=t_indexes.raw_data,
                    role={
                        col: t_indexes.roles.get(col, InfoRole())
                        for col in t_indexes.columns
                    },
                )

            if full_data is None:
                full_data = experiment_data.ds.merge(
                    reindexed_matched,
                    left_index=True,
                    right_index=True,
                    how="left",
                )
            else:
                full_data = full_data.merge(
                    reindexed_matched,
                    left_index=True,
                    right_index=True,
                    how="left",
                )
            full_data.checkpoint(eager=True)

        self.full_data = full_data if full_data is not None else experiment_data.ds

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

    @staticmethod
    def _collect_grouped_indexes(
        experiment_data: ExperimentData, group: dict
    ) -> Dataset:
        """Collect matched indexes for grouped matching results.

        Args:
            experiment_data: The experiment data container.
            group: Dictionary mapping group names to matched index strings.

        Returns:
            Dataset with collected indexes sorted by index.
        """
        group_indexes_id = experiment_data.ds.search_columns(GroupingRole())
        indexes = []
        for group_name, values in group.items():
            ds = SmallDataset.from_dict(
                {
                    "indexes": list(
                        map(int, values.split(MATCHING_INDEXES_SPLITTER_SYMBOL))
                    )
                },
                roles={"indexes": StatisticRole()},
            )
            ds.index = Adapter.to_list(
                experiment_data.ds[
                    experiment_data.ds[group_indexes_id] == group_name
                ].index
            )
            indexes.append(ds)
        return indexes[0].append(indexes[1:]).sort()

    @staticmethod
    def _get_spark_indexes(experiment_data: ExperimentData) -> Dataset:
        """Select matched-index columns lazily from the experiment dataset.

        ``FaissNearestNeighbors`` stores matched indices as
        ``AdditionalMatchingRole`` columns merged directly into
        ``experiment_data.ds`` (see ``ExperimentData._set_additional_fields``),
        already aligned to the dataset index with ``-1`` for unmatched
        observations. Selecting these columns is a lazy Spark
        transformation — no data is collected to the driver.

        Args:
            experiment_data: The experiment data container.

        Returns:
            A lazy Spark-backed ``Dataset`` with one ``indexes_{i}`` column
            per neighbor position, or an empty Dataset when no matched
            indices were stored.
        """
        ids = experiment_data.get_ids(
            FaissNearestNeighbors, ExperimentDataEnum.additional_fields
        )[FaissNearestNeighbors.__name__][ExperimentDataEnum.additional_fields.value]
        additional = experiment_data.additional_fields
        available = sorted(
            (col for col in ids if col in additional.columns),
            key=lambda c: int(str(c).split(ID_SPLIT_SYMBOL)[-1]),
        )
        if not available:
            return Dataset.create_empty(
                roles={},
                backend=BackendsEnum.spark,
                session=experiment_data.ds.session,
            )
        indexes = additional[available]
        return indexes.rename(
            {
                col: f"indexes_{col.split(ID_SPLIT_SYMBOL)[-1]}"
                for col in indexes.columns
            }
        )

    def _extract_driver_indexes(
        self,
        experiment_data: ExperimentData,
        reformatted_summary: dict[str, Any],
    ) -> Dataset | SmallDataset:
        """Parse matched indexes from the summary string on the driver.

        Legacy extraction path used only for the Pandas backend, where the
        data already resides in driver memory. Handles three branches of
        index extraction:

        1. Nested grouped indexes (``are_nested=True``): alignment is done
           inside ``_collect_grouped_indexes`` per group.
        2. Flat grouped indexes (``are_nested=False``): alignment to the
           original dataset index is applied here.
        3. Single (non-grouped) indexes: alignment with a length-mismatch
           guard.

        Args:
            experiment_data: The experiment data container with matching
                results stored in ``analysis_tables`` and ``variables``.
            reformatted_summary: Flat summary dictionary regrouped by
                ``_reformat_summary``. The ``indexes`` entry is popped from
                it as a side effect.

        Returns:
            A ``Dataset`` or ``SmallDataset`` with matched index columns,
            aligned to the original dataset index when lengths match.
            Returns an empty ``SmallDataset`` when no indexes are found.
        """
        ds_len = len(experiment_data.ds)

        if "indexes" in reformatted_summary.keys():
            indexes_items = reformatted_summary.pop("indexes")
            are_nested = all(isinstance(v, dict) for v in indexes_items.values())

            if are_nested:
                # ── Branch 1: nested grouped indexes ──────────────────
                # Alignment is handled inside _collect_grouped_indexes
                # via filtering the original ds by group mask.
                index_parts = [
                    self._collect_grouped_indexes(experiment_data, values).rename(
                        {"indexes": f"indexes_{group}"}
                    )
                    for group, values in indexes_items.items()
                ]
            else:
                # ── Branch 2: flat grouped indexes ────────────────────
                # Values are strings; SmallDataset.from_dict creates a
                # RangeIndex.  Must align to the original ds index.
                index_parts = []
                for group, values in indexes_items.items():
                    idx_values = list(
                        map(int, values.split(MATCHING_INDEXES_SPLITTER_SYMBOL))
                    )
                    ds = SmallDataset.from_dict(
                        {f"indexes_{group}": idx_values},
                        roles={f"indexes_{group}": StatisticRole()},
                    )
                    if len(ds) == ds_len:
                        ds.index = Adapter.to_list(experiment_data.ds.index)
                    else:
                        warnings.warn(
                            f"Matched indexes length ({len(ds)}) != "
                            f"dataset length ({ds_len}) for group "
                            f"'{group}'. Alignment skipped.",
                            UserWarning,
                            stacklevel=2,
                        )
                    index_parts.append(ds)

            if index_parts:
                indexes = index_parts[0].append(other=index_parts[1:], axis=1).sort()
            else:
                indexes = SmallDataset.create_empty()

        else:
            # ── Branch 3: single (non-grouped) indexes ────────────────
            indexes_data = self.summary.get("indexes", "").split(
                MATCHING_INDEXES_SPLITTER_SYMBOL
            )
            if indexes_data and indexes_data[0]:
                indexes = SmallDataset.from_dict(
                    {"indexes": list(map(int, indexes_data))},
                    roles={"indexes": AdditionalMatchingRole()},
                )
                if len(indexes) == ds_len:
                    # The matched indexes are already on the driver (parsed
                    # from the summary string).  Collecting the dataset index
                    # costs the same, so a single code path is kept here.
                    indexes.index = Adapter.to_list(experiment_data.ds.index)
                else:
                    warnings.warn(
                        f"Matched indexes length ({len(indexes)}) != "
                        f"dataset length ({ds_len}). Alignment skipped.",
                        UserWarning,
                        stacklevel=2,
                    )
            else:
                indexes = SmallDataset.create_empty()

        return indexes

    def _restore_index_role(self, experiment_data: ExperimentData) -> None:
        index_fields = experiment_data.ds.search_columns(IndexRole())
        if not index_fields:
            return
        index_field = index_fields[0]

        if not self.indexes.is_empty():
            index_data = experiment_data.ds[index_field]
            mapped_indexes = self.indexes.map_index_values(index_data)
            index_role = experiment_data.ds.roles[index_field]
            self.indexes = Dataset(
                roles={
                    column: AdditionalMatchingRole(index_role.data_type)
                    for column in mapped_indexes.columns
                },
                data=mapped_indexes.raw_data,
                session=experiment_data.ds.session,
            )
        if not self.full_data.is_empty():
            self.full_data = self.full_data.set_index(index_field, drop=False)

    def extract(self, experiment_data: ExperimentData) -> None:
        """Extract and format all matching results from experiment data.

        For the Spark backend, matched indexes are taken directly from the
        lazy ``additional_fields`` columns of ``experiment_data.ds`` — the
        summary-string round trip and any driver-side index collection are
        skipped entirely. For the Pandas backend, the legacy string-based
        extraction is preserved.

        Args:
            experiment_data: The experiment data container with matching
                results stored in ``analysis_tables`` and ``variables``.
        """
        # Let the base class handle additional_reporters (like quality_results)
        super().extract(experiment_data)

        reformatted_summary = self._reformat_summary(self.summary)

        if self.compute_indexes:
            if experiment_data.ds.backend_type == BackendsEnum.spark:
                reformatted_summary.pop("indexes", None)
                indexes = self._get_spark_indexes(experiment_data)
            else:
                indexes = self._extract_driver_indexes(
                    experiment_data, reformatted_summary
                )
        else:
            reformatted_summary.pop("indexes", None)
            indexes = SmallDataset.create_empty()
            logger.debug("Skipping indexes extraction (compute_indexes=False).")

        # ── Build summary table from remaining metrics ─────────────────
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
            self._extract_full_data(experiment_data, indexes)
        else:
            self.full_data = Dataset.create_empty(
                roles={},
                backend=experiment_data.ds.backend_type,
                session=experiment_data.ds.session,
            )
            self.indexes = (
                indexes
                if self.compute_indexes
                else Dataset.create_empty(
                    roles={},
                    backend=experiment_data.ds.backend_type,
                    session=experiment_data.ds.session,
                )
            )
            logger.debug("Skipping full_data extraction (extract_full_data=False).")
        self._restore_index_role(experiment_data)
        self.summary.raw_data = self.summary.raw_data.round(4)
